"""优化建议模块。

读取某个 run 的评测结果，由"建议模型"诊断当前 System Prompt 的短板，
产出诊断分析 + 修订版 System Prompt，写入 run 目录下的 advise.md。

设计原则：最大化复用 reporter 的 load/extract 工具与 models 的调用机制，
不重写读取与调用逻辑。
"""

import json
from pathlib import Path

from src.config import AdviseConfig
from src.constants import RESULTS_DIR
from src.experiment import load_run_meta
from src.models import call_model, create_client
from src.reporter import (
    _extract_response_text,
    load_responses,
    load_scores,
)


def _select_low_score_samples(
    responses: list[dict],
    scores: dict[int, dict],
    cfg: AdviseConfig,
) -> list[dict]:
    """选取喂给建议模型的低分样本。

    策略：
    - overall <= low_score_threshold 视为低分
    - 低分样本不足 min_low_samples 时，按 overall 升序补足
    - 总数上限 max_samples 截断

    Returns:
        list[dict]，每条含 row_index / query / response / scores(各维度) / analysis，
        按 overall 升序排列（最低分在前）。
    """
    # 过滤掉无 overall 分或无对应 response 的行
    candidates = []
    for r in responses:
        idx = r["row_index"]
        score = scores.get(idx)
        if not score or "overall" not in score:
            continue
        overall = score["overall"]
        if not isinstance(overall, (int, float)):
            continue
        candidates.append({
            "row_index": idx,
            "query": r.get("query", ""),
            "response": _extract_response_text(r.get("response", {})),
            "overall": overall,
            "scores": {
                k: v for k, v in score.items()
                if k not in ("row_index", "query", "response_summary", "error", "analysis")
            },
            "analysis": score.get("analysis", ""),
        })

    # 按 overall 升序
    candidates.sort(key=lambda x: x["overall"])

    # 阈值低分
    low = [c for c in candidates if c["overall"] <= cfg.low_score_threshold]

    # 不足则补足
    if len(low) < cfg.min_low_samples:
        # 从候选里取尚未入选的，继续按升序补
        low_ids = {c["row_index"] for c in low}
        for c in candidates:
            if len(low) >= cfg.min_low_samples:
                break
            if c["row_index"] not in low_ids:
                low.append(c)
                low_ids.add(c["row_index"])
        # 补足后重新按 overall 排序，保持最低分在前
        low.sort(key=lambda x: x["overall"])

    # 上限截断
    return low[: cfg.max_samples]


def _format_summary(summary: dict | None, dimensions: list[str]) -> str:
    """把 summary.json 的各维度平均分格式化为 markdown。"""
    if not summary:
        return "（无 summary.json，统计缺失）"
    stats = summary.get("summary", {})
    lines = [f"- 总样本数：{stats.get('total_items', 'N/A')}"]
    for dim in dimensions:
        key = f"avg_{dim}"
        if key in stats:
            lines.append(f"- {dim}：{stats[key]}")
    return "\n".join(lines)


def _build_advise_user_content(
    current_prompt: str,
    summary: dict | None,
    dimensions: list[str],
    samples: list[dict],
) -> str:
    """构造建议模型的 user 消息（markdown）。"""
    parts: list[str] = []

    parts.append("## 当前 System Prompt\n")
    parts.append("```")
    parts.append(current_prompt or "（无 system prompt，meta.json 缺失 prompt_content）")
    parts.append("```\n")

    parts.append("## 评分统计\n")
    parts.append(_format_summary(summary, dimensions))
    parts.append("\n")

    parts.append(f"## 低分样本（共 {len(samples)} 条）\n")
    for i, s in enumerate(samples, 1):
        parts.append(f"### 样本 {i}（row_index={s['row_index']}，overall={s['overall']}）")
        parts.append(f"- query：{s['query']}")
        score_line = ", ".join(f"{k}={v}" for k, v in s["scores"].items())
        parts.append(f"- 各维度分数：{score_line}")
        parts.append(f"- response：\n\n{s['response']}")
        parts.append(f"- 评测分析：\n\n{s['analysis']}")
        parts.append("")

    return "\n".join(parts)


async def run_advise(run_name: str, advise_cfg: AdviseConfig) -> Path:
    """读取 run 结果 → 选低分样本 → 构造提示 → 调用建议模型 → 写 advise.md。

    Args:
        run_name: 实验运行名称（须已评测，即 scores.jsonl 存在）
        advise_cfg: 建议配置

    Returns:
        生成的 advise.md 路径
    """
    result_dir = RESULTS_DIR / run_name
    responses_path = result_dir / "responses.jsonl"
    scores_path = result_dir / "scores.jsonl"
    summary_path = result_dir / "summary.json"

    # 前置检查：必须已评测
    if not scores_path.exists():
        raise FileNotFoundError(
            f"该 run 尚未评测（{scores_path} 不存在），请先运行：python -m src.cli eval {run_name}"
        )

    # 1. 读 run 产物
    responses = load_responses(responses_path)
    scores = load_scores(scores_path)

    summary = None
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    dimensions = summary.get("dimensions", []) if summary else []

    # 当前 system prompt 从 meta.json 取（保留 dataset_prompt_template 注入前的原始模板）
    meta = load_run_meta(run_name)
    current_prompt = meta.get("prompt_content", "")

    # 2. 选低分样本
    samples = _select_low_score_samples(responses, scores, advise_cfg)

    # 3. 构造 messages
    user_content = _build_advise_user_content(current_prompt, summary, dimensions, samples)
    messages = [
        {"role": "system", "content": advise_cfg.prompt},
        {"role": "user", "content": user_content},
    ]

    # 4. 调用建议模型
    client = create_client(advise_cfg.model)
    response, _ = await call_model(client, advise_cfg.model, messages)

    # 5. 写产物
    advice_text = _extract_response_text(response)
    advise_path = result_dir / "advise.md"
    advise_path.write_text(advice_text, encoding="utf-8")

    return advise_path
