"""评测模块（LLM-as-Judge）。

读取 responses.jsonl，用 Judge 模型对每条结果打分。
逐条评分写入 scores.jsonl，汇总统计写入 summary.json。
"""

import asyncio
import json
import re
from pathlib import Path

from tqdm import tqdm

from src.config import ModelConfig
from src.constants import RESULTS_DIR
from src.experiment import load_run_meta
from src.models import create_client, call_model, call_model_stream
from src.reporter import load_responses, load_scores

JUDGE_SEED = 7
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


async def run_evaluation(run_name: str):
    """对实验结果进行 LLM-as-Judge 评测。"""
    meta = load_run_meta(run_name)
    if not meta.get("judge"):
        print(f"[skip] run '{run_name}' has no judge config in meta.json")
        return

    judge_data = meta["judge"]
    judge_model_cfg = ModelConfig(**judge_data["model"])
    client = create_client(judge_model_cfg)

    responses_path = RESULTS_DIR / run_name / "responses.jsonl"
    if not responses_path.exists():
        print(f"[error] no results found at {responses_path}")
        return

    results = load_responses(responses_path)
    if not results:
        print(f"[error] no valid results found in {responses_path}")
        return

    scores_path = RESULTS_DIR / run_name / "scores.jsonl"
    summary_path = RESULTS_DIR / run_name / "summary.json"

    # 加载已有评测结果，支持断点续评
    existing_scores = load_scores(scores_path)
    done_count = len(existing_scores)
    if done_count > 0:
        print(f"[resume] {done_count}/{len(results)} scores already exist, resuming...")

    # Judge prompt 作为 system 消息（评分标准不变，user 消息每次不同）
    system_prompt = judge_data["prompt"]
    pbar = tqdm(total=len(results), initial=done_count, desc=f"eval/{run_name}", unit="item")

    for r in results:
        row_idx = r["row_index"]
        if row_idx in existing_scores:
            continue

        # 提取回复文本和上下文信息
        response_text = _extract_response_text(r["response"])
        messages = _build_judge_messages(
            system_prompt, r["query"], response_text,
            language=r.get("language"), location=r.get("location"),
        )

        score_entry = None
        dims = judge_data["dimensions"]
        for attempt in range(1 + MAX_RETRIES):
            try:
                request = {"messages": messages, "seed": JUDGE_SEED}
                if judge_model_cfg.stream:
                    response_dict, _, _ = await call_model_stream(client, judge_model_cfg, request)
                else:
                    response_dict, _ = await call_model(client, judge_model_cfg, request)
                content = response_dict["choices"][0]["message"]["content"] or ""
                parsed = _parse_judge_output(content, dims)
                score_entry = {"row_index": row_idx, "query": r["query"], "response_summary": response_text[:200],
                               **parsed}
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    tqdm.write(f"[warn] row {row_idx} parse failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(delay)
                else:
                    tqdm.write(f"[error] judge failed for row {row_idx}: {e}")
                    score_entry = {"row_index": row_idx, "error": str(e)}

        if score_entry is not None:
            existing_scores[row_idx] = score_entry
            _append_score(scores_path, score_entry)
        pbar.update(1)

    pbar.close()
    # 汇总统计
    all_scores = [existing_scores[i] for i in sorted(existing_scores)]
    summary = _compute_summary(all_scores, judge_data["dimensions"])

    output = {
        "experiment": run_name,
        "judge_model": judge_model_cfg.model,
        "dimensions": judge_data["dimensions"],
        "summary": summary,
        "total_scored": len(all_scores),
    }
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] evaluation saved → {summary_path}")
    print(f"  Summary: {json.dumps(summary, ensure_ascii=False)}")


def _append_score(scores_path: Path, entry: dict):
    """追加一条评分到 scores.jsonl。"""
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_judge_messages(
    system_prompt: str, query: str, response: str,
    language: str | None = None, location: str | None = None,
) -> list[dict]:
    """构建 judge API 调用的 messages。

    Args:
        system_prompt: 完整的评分标准（来自 judge prompt 文件）
        query: 用户问题
        response: 模型回复
        language: 用户语言，用于评测本地化维度
        location: 用户位置，用于评测本地化维度
    """
    user_content = f"## 待评测内容\n\n**用户问题：**\n{query}\n\n**AI 回复：**\n{response}"

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
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return str(response)


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

    return parsed


def _compute_summary(scores: list[dict], dimensions: list[str]) -> dict:
    """计算各维度平均分。"""
    summary = {"total_items": len(scores)}

    for dim in dimensions:
        values = [s[dim] for s in scores if dim in s and isinstance(s[dim], (int, float))]
        if values:
            summary[f"avg_{dim}"] = round(sum(values) / len(values), 2)
    return summary
