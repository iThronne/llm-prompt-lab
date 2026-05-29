"""评测模块（LLM-as-Judge）。

读取 results.jsonl，用 Judge 模型对每条结果打分。
输出 evaluation.json 包含每条评分和汇总统计。
"""

import json
import asyncio
import time
from pathlib import Path

from jinja2 import Template
from tqdm import tqdm

from src.config import Config
from src.models import create_client


RESULTS_DIR = Path("results")


async def run_evaluation(config: Config, experiment_name: str):
    """对实验结果进行 LLM-as-Judge 评测。"""
    exp = config.get_experiment(experiment_name)
    if not exp.judge:
        print(f"[skip] experiment '{experiment_name}' has no judge config")
        return

    judge_cfg = config.get_model(exp.judge.model)
    client = create_client(judge_cfg)

    results_path = RESULTS_DIR / experiment_name / "results.jsonl"
    if not results_path.exists():
        print(f"[error] no results found at {results_path}")
        return

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    eval_path = RESULTS_DIR / experiment_name / "evaluation.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有评测结果，支持断点续评
    existing_scores: dict[int, dict] = {}
    if eval_path.exists():
        existing = json.loads(eval_path.read_text(encoding="utf-8"))
        for item in existing.get("scores", []):
            existing_scores[item["row_index"]] = item

    judge_template = Template(exp.judge.prompt)
    scores = []
    pbar = tqdm(results, desc=f"eval/{experiment_name}", unit="item")

    for r in pbar:
        row_idx = r["row_index"]
        if row_idx in existing_scores:
            scores.append(existing_scores[row_idx])
            continue

        # 提取回复文本
        response_text = _extract_response_text(r["response"])
        judge_prompt = judge_template.render(query=r["query"], response=response_text)

        for attempt in range(3):
            try:
                judge_resp = await client.chat.completions.create(
                    model=judge_cfg.model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0,
                )
                content = judge_resp.choices[0].message.content or ""
                parsed = _parse_judge_output(content)
                score_entry = {"row_index": row_idx, "query": r["query"], "response_summary": response_text[:200], **parsed}
                scores.append(score_entry)
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    tqdm.write(f"[error] judge failed for row {row_idx}: {e}")
                    scores.append({"row_index": row_idx, "error": str(e)})

    # 汇总统计
    summary = _compute_summary(scores, exp.judge.dimensions)

    output = {"experiment": experiment_name, "judge_model": exp.judge.model, "dimensions": exp.judge.dimensions, "summary": summary, "scores": scores}
    eval_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] evaluation saved → {eval_path}")
    print(f"  Summary: {json.dumps(summary, ensure_ascii=False)}")


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