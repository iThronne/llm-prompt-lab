"""评测模块（LLM-as-Judge）。

读取 responses.jsonl，用 Judge 模型对每条结果打分。
逐条评分写入 scores.jsonl，汇总统计写入 summary.json。
"""

import asyncio
import json
from pathlib import Path

from tqdm import tqdm

from src.config import ModelConfig
from src.constants import RESULTS_DIR
from src.experiment import load_run_meta
from src.models import create_client

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

    results = _load_responses(responses_path)
    if not results:
        print(f"[error] no valid results found in {responses_path}")
        return

    scores_path = RESULTS_DIR / run_name / "scores.jsonl"
    summary_path = RESULTS_DIR / run_name / "summary.json"

    # 加载已有评测结果，支持断点续评
    existing_scores = _load_scores(scores_path)
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

        # 提取回复文本
        response_text = _extract_response_text(r["response"])
        messages = _build_judge_messages(system_prompt, r["query"], response_text)

        score_entry = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                judge_resp = await client.chat.completions.create(
                    model=judge_model_cfg.model,
                    messages=messages,
                    temperature=0,
                    seed=JUDGE_SEED,
                )
                content = judge_resp.choices[0].message.content or ""
                parsed = _parse_judge_output(content)
                score_entry = {"row_index": row_idx, "query": r["query"], "response_summary": response_text[:200],
                               **parsed}
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
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


def _load_responses(path: Path) -> list[dict]:
    """从 responses.jsonl 加载结果，按 row_index 去重（保留最后一条）。"""
    responses_by_idx: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                responses_by_idx[entry["row_index"]] = entry
    return [responses_by_idx[i] for i in sorted(responses_by_idx)]


def _load_scores(scores_path: Path) -> dict[int, dict]:
    """从 scores.jsonl 加载已有评分，用于断点续评。"""
    scores: dict[int, dict] = {}
    if not scores_path.exists():
        return scores
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                scores[entry["row_index"]] = entry
    return scores


def _append_score(scores_path: Path, entry: dict):
    """追加一条评分到 scores.jsonl。"""
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_judge_messages(system_prompt: str, query: str, response: str) -> list[dict]:
    """构建 judge API 调用的 messages。

    Args:
        system_prompt: 完整的评分标准（来自 judge prompt 文件）
        query: 用户问题
        response: 模型回复
    """
    user_content = f"## 待评测内容\n\n**用户问题：**\n{query}\n\n**AI 回复：**\n{response}"
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


def _parse_judge_output(content: str) -> dict:
    """解析 Judge 模型输出的 JSON。"""
    content = content.strip()
    # 尝试提取 JSON 块（可能被 markdown 代码块包裹）
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_output": content}


def _compute_summary(scores: list[dict], dimensions: list[str]) -> dict:
    """计算各维度平均分。"""
    summary = {"total_items": len(scores)}

    for dim in dimensions:
        values = [s[dim] for s in scores if dim in s and isinstance(s[dim], (int, float))]
        if values:
            summary[f"avg_{dim}"] = round(sum(values) / len(values), 2)
    return summary
