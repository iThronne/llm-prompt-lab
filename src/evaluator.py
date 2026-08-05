"""评测模块（LLM-as-Judge）。

读取 responses.jsonl，用 Judge 模型对每条结果打分。
逐条评分写入 scores.jsonl，汇总统计写入 summary.json。

评测配置从 eval.yaml 实时加载（不依赖 run 时的 meta.json 快照），
通过 eval_meta.json 记录当次评测使用的配置 hash，
实现断点续评（hash 相同）和配置变更检测（hash 不同）。
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path

from tqdm import tqdm

from src.config import EvalConfig
from src.constants import RESULTS_DIR
from src.models import create_client, call_model, call_model_stream
from src.reporter import load_responses, load_scores
from src.sanitizer import compile_rules, sanitize_messages

JUDGE_SEED = 7
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds
EVAL_META_FILE = "eval_meta.json"


def compute_eval_hash(eval_cfg: EvalConfig) -> str:
    """计算评测配置的 hash，用于检测配置变更。"""
    payload = {
        "model": eval_cfg.model.model_dump(),
        "prompt": eval_cfg.prompt,
        "dimensions": eval_cfg.dimensions,
        "sanitize": [r.model_dump() for r in eval_cfg.sanitize],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


async def run_evaluation(
    run_name: str,
    eval_cfg: EvalConfig,
    concurrency: int = 1,
    force: bool = False,
):
    """对实验结果进行 LLM-as-Judge 评测。

    Args:
        run_name: 实验 run 名称
        eval_cfg: 评测配置（从 eval.yaml 加载）
        concurrency: 并发评测数，默认 1（串行）
        force: 配置变更时是否强制重新评测
    """
    judge_model_cfg = eval_cfg.model
    client = create_client(judge_model_cfg)
    compiled_sanitize = compile_rules(eval_cfg.sanitize)

    if eval_cfg.sanitize:
        print(f"[info] sanitize rules loaded: {len(eval_cfg.sanitize)} rule(s)")

    responses_path = RESULTS_DIR / run_name / "responses.jsonl"
    if not responses_path.exists():
        print(f"[error] no results found at {responses_path}")
        return

    results = load_responses(responses_path)
    if not results:
        print(f"[error] no valid results found in {responses_path}")
        return

    result_dir = RESULTS_DIR / run_name
    scores_path = result_dir / "scores.jsonl"
    summary_path = result_dir / "summary.json"
    eval_meta_path = result_dir / EVAL_META_FILE

    # 计算当前评测配置的 hash
    current_hash = compute_eval_hash(eval_cfg)

    # 检查已有评测结果和配置 hash
    existing_scores = load_scores(scores_path)
    if existing_scores:
        old_hash = None
        if eval_meta_path.exists():
            old_meta = json.loads(eval_meta_path.read_text(encoding="utf-8"))
            old_hash = old_meta.get("eval_config_hash")

        if old_hash and old_hash != current_hash:
            if not force:
                print(f"[warn] 评测配置已变更 (hash: {old_hash} → {current_hash})")
                print(f"  已有 {len(existing_scores)} 条评分结果，当前配置与上次不同。")
                print(f"  如需重新评测，请先手动删除以下文件后重试：")
                print(f"    {scores_path}")
                print(f"    {eval_meta_path}")
                print(f"    {summary_path}")
                print(f"  或使用 --force 强制覆盖。")
                return
            else:
                print(f"[force] 评测配置已变更 (hash: {old_hash} → {current_hash})，清空旧结果")
                for f in (scores_path, eval_meta_path, summary_path):
                    if f.exists():
                        f.unlink()
                existing_scores = {}
        elif not old_hash:
            # 旧的评测结果（无 eval_meta.json），当作配置未变处理，允许续评
            print(f"[info] 发现旧评测结果（无 eval_meta.json），以当前配置续评")
        else:
            # hash 相同，正常断点续评
            done_count = len(existing_scores)
            print(f"[resume] {done_count}/{len(results)} scores already exist, resuming...")

    # 空回复没有可评内容：记录为已跳过，避免送给 Judge，也避免断点续跑时反复尝试。
    # 同时覆盖旧版本可能已经为这些行生成的分数，防止其继续污染统计。
    empty_response_count = 0
    for r in results:
        if _is_empty_response(r.get("response")):
            row_idx = r["row_index"]
            existing_scores[row_idx] = _build_skipped_score(r, eval_cfg.dimensions)
            empty_response_count += 1
    if empty_response_count:
        print(f"[info] 跳过 {empty_response_count} 条空回复（不调用 Judge、不计入统计）")

    # 筛选待评测的条目
    pending = [r for r in results if r["row_index"] not in existing_scores]

    if pending:
        system_prompt = eval_cfg.prompt
        dims = eval_cfg.dimensions
        semaphore = asyncio.Semaphore(concurrency)
        write_lock = asyncio.Lock()
        pbar = tqdm(total=len(results), initial=len(existing_scores), desc=f"eval/{run_name}", unit="item")

        if concurrency > 1:
            print(f"[info] 并发评测: concurrency={concurrency}")

        # 首样本调试：把 judge 的完整入参 dump 到 results 目录，方便核对脱敏 / JSON 形态。
        # 评测完一轮可手动删除该文件，或通过环境变量关闭。
        sample_dumped = {"done": False}
        sample_dump_path = result_dir / "_judge_input_sample.json"

        async def _evaluate_one(r: dict):
            """评测单条数据，带并发控制和重试。"""
            async with semaphore:
                row_idx = r["row_index"]
                response_text = _extract_response_text(r["response"])

                # 取 rendered_request 中的完整 messages（含 system）做脱敏后传给 judge
                rendered_request = r.get("rendered_request", {})
                all_messages = rendered_request.get("messages", [])
                sanitized_messages = sanitize_messages(all_messages, compiled_sanitize)

                messages = _build_judge_messages(
                    system_prompt, sanitized_messages, response_text,
                    language=r.get("language"), location=r.get("location"),
                )

                # 首样本调试输出（只 dump 第一条，避免刷屏）
                if not sample_dumped["done"]:
                    async with write_lock:
                        if not sample_dumped["done"]:
                            sample_dumped["done"] = True
                            try:
                                sample_dump_path.write_text(
                                    json.dumps(
                                        {"row_index": row_idx, "judge_messages": messages},
                                        ensure_ascii=False, indent=2,
                                    ),
                                    encoding="utf-8",
                                )
                                preview = messages[1]["content"]
                                tqdm.write(
                                    f"[debug] judge 入参样本已写入 {sample_dump_path}\n"
                                    f"  row {row_idx} user content 前 500 字:\n"
                                    f"  {preview[:500]}{'...' if len(preview) > 500 else ''}"
                                )
                            except Exception as e:
                                tqdm.write(f"[warn] 首样本调试 dump 失败: {e}")

                score_entry = None
                # 通过 model_copy 注入 seed 参数（保证评测可复现）
                seeded_cfg = judge_model_cfg.model_copy(update={"seed": JUDGE_SEED})
                for attempt in range(1 + MAX_RETRIES):
                    try:
                        if judge_model_cfg.stream:
                            response_dict, _, _ = await call_model_stream(client, seeded_cfg, messages)
                        else:
                            response_dict, _ = await call_model(client, seeded_cfg, messages)

                        finish_reason = response_dict["choices"][0].get("finish_reason")
                        if finish_reason == "length":
                            raise ValueError("输出被截断 (finish_reason=length)，JSON 不完整")

                        content = response_dict["choices"][0]["message"]["content"] or ""
                        parsed = _parse_judge_output(content, dims)
                        score_entry = {
                            "row_index": row_idx, "query": r["query"],
                            "response_summary": response_text[:200], **parsed,
                        }
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES:
                            delay = RETRY_BASE_DELAY * (2 ** attempt)
                            tqdm.write(f"[warn] row {row_idx} parse failed (attempt {attempt + 1}), retrying: {e}")
                            await asyncio.sleep(delay)
                        else:
                            tqdm.write(f"[error] judge failed for row {row_idx}: {e}")
                            score_entry = None

                if score_entry is not None:
                    async with write_lock:
                        existing_scores[row_idx] = score_entry
                        _append_score(scores_path, score_entry)
                pbar.update(1)

        await asyncio.gather(*[_evaluate_one(r) for r in pending])
        pbar.close()

    # 按 row_index 排序后重写 scores.jsonl
    _sort_scores(scores_path, existing_scores)

    # 写入 eval_meta.json（记录本次评测的配置快照）
    eval_meta = {
        "eval_config_hash": current_hash,
        "judge": eval_cfg.model_dump(),
    }
    eval_meta_path.write_text(
        json.dumps(eval_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 汇总统计
    all_scores = [existing_scores[i] for i in sorted(existing_scores)]
    summary = _compute_summary(all_scores, eval_cfg.dimensions)
    scored_count = sum(_has_any_numeric_score(s, eval_cfg.dimensions) for s in all_scores)
    skipped_count = sum(s.get("error") == "empty_response" for s in all_scores)
    failed_count = len(results) - len(all_scores)

    output = {
        "experiment": run_name,
        "judge_model": judge_model_cfg.model,
        "dimensions": eval_cfg.dimensions,
        "summary": summary,
        "total_scored": scored_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
    }
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] evaluation saved → {summary_path}")
    if failed_count > 0:
        print(f"  {failed_count} item(s) failed, run eval again to retry")
    print(f"  Summary: {json.dumps(summary, ensure_ascii=False)}")


def _append_score(scores_path: Path, entry: dict):
    """追加一条评分到 scores.jsonl。"""
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _sort_scores(scores_path: Path, existing_scores: dict):
    """按 row_index 排序后重写 scores.jsonl。"""
    if not scores_path.exists() or not existing_scores:
        return

    with open(scores_path, "w", encoding="utf-8") as f:
        for row_idx in sorted(existing_scores):
            f.write(json.dumps(existing_scores[row_idx], ensure_ascii=False) + "\n")


def _build_judge_messages(
    system_prompt: str, candidate_messages: list[dict], response: str,
    language: str | None = None, location: str | None = None,
) -> list[dict]:
    """构建 judge API 调用的 messages。

    Args:
        system_prompt: 完整的评分标准（来自 judge prompt 文件，已包含体裁清单）
        candidate_messages: 候选模型的完整 messages 列表（已脱敏，含 system）
        response: 模型回复
        language: 用户语言，用于评测本地化维度
        location: 用户位置，用于评测本地化维度
    """
    # 候选模型完整请求（脱敏后）以 JSON 形式注入，保留原始 role/content 结构
    messages_json = json.dumps(candidate_messages, ensure_ascii=False, indent=2)
    user_content = (
        f"## 候选模型完整请求（已脱敏）\n\n"
        f"```json\n{messages_json}\n```\n\n"
        f"## 候选模型输出\n\n{response}"
    )

    # 在待评测内容前添加用户上下文（语言、位置）
    context_parts = []
    if language:
        context_parts.append(f"语言：{language}")
    if location:
        context_parts.append(f"位置：{location}")
    if context_parts:
        user_content = "**用户上下文：**\n" + "\n".join(context_parts) + "\n\n" + user_content

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _extract_response_text(response: dict) -> str:
    """从 API 响应中提取回复文本。"""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _is_empty_response(response: dict | None) -> bool:
    """判断候选模型回复是否为 None、缺失或仅含空白字符。"""
    return not _extract_response_text(response).strip()


def _build_skipped_score(result: dict, dimensions: list[str]) -> dict:
    """为空回复生成可断点续跑、但不会参与统计的占位评分。"""
    return {
        "row_index": result["row_index"],
        "query": result.get("query", ""),
        "response_summary": "",
        "error": "empty_response",
        "analysis": "候选模型回复为空，已跳过评测。",
        **{dim: None for dim in dimensions},
    }


def _extract_json_string(content: str) -> str:
    """从模型输出中提取 JSON 字符串，尽可能容错。

    按优先级尝试：
    1. markdown ```json``` 代码块
    2. 首个 { 到末尾匹配的 }
    3. 正则匹配最外层 JSON 对象
    """
    # 1. markdown 代码块
    if "```json" in content:
        return content.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in content:
        inner = content.split("```", 2)
        if len(inner) >= 3:
            return inner[1].strip()

    # 2. 找到第一个 { 和最后一个 }，截取中间部分
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return content[first_brace:last_brace + 1].strip()

    return content.strip()


def _fix_common_json_issues(s: str) -> str:
    """修复常见的 JSON 格式问题。"""
    # 去除尾部多余的逗号（如 {"a":1,}）
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # 修复字符串值中的未转义换行（analysis 字段常见）
    # 策略：在引号内将裸换行替换为 \n
    result = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            pass  # skip \r inside strings
        elif in_string and ch == "\t":
            result.append("\\t")
        elif in_string and ch == '"':
            # 未转义的引号 — 不应该到这里，但防御性处理
            result.append('\\"')
        else:
            result.append(ch)
    return "".join(result)


def _parse_judge_output(content: str, dimensions: list[str]) -> dict:
    """解析 Judge 模型输出的 JSON。

    Args:
        content: Judge 模型的原始输出
        dimensions: 必需的评分维度列表

    Raises:
        ValueError: JSON 解析失败或缺少必需字段
    """
    extracted = _extract_json_string(content)

    # 先尝试直接解析
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        # 尝试修复常见问题后再解析
        fixed = _fix_common_json_issues(extracted)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 Judge 输出为 JSON: {e} | 内容: {content[:300]}")

    # 校验必需字段：analysis + 所有评分维度
    missing = []
    if "analysis" not in parsed:
        missing.append("analysis")
    for dim in dimensions:
        if dim not in parsed:
            missing.append(dim)
    if missing:
        raise ValueError(f"Judge 输出缺少必需字段: {missing}")

    # 不适用维度使用 null；兼容模型偶尔按展示约定返回的 "-"。
    for dim in dimensions:
        value = parsed[dim]
        if value == "-":
            parsed[dim] = None
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError(f"Judge 输出维度 {dim} 的值无效: {value!r}（应为 1-5 或 null）")

    return parsed


def _has_numeric_score(score: dict, dimension: str) -> bool:
    """是否包含一个可参与统计的数值评分（bool 不视为分数）。"""
    value = score.get(dimension)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _has_any_numeric_score(score: dict, dimensions: list[str]) -> bool:
    """一条记录是否至少有一个配置维度可参与统计。"""
    return any(_has_numeric_score(score, dim) for dim in dimensions)


def _compute_summary(scores: list[dict], dimensions: list[str]) -> dict:
    """按维度计算平均分；空回复和不适用维度不进入对应分母。"""
    summary = {
        "total_items": sum(_has_any_numeric_score(s, dimensions) for s in scores),
        "total_records": len(scores),
    }

    for dim in dimensions:
        values = [s[dim] for s in scores if _has_numeric_score(s, dim)]
        summary[f"count_{dim}"] = len(values)
        if values:
            summary[f"avg_{dim}"] = round(sum(values) / len(values), 2)
    return summary
