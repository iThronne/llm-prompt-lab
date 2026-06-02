"""报告生成模块。

提供实验结果的可视化报告：
- HTML 报告：自包含 HTML 文件，内置图表 + 可交互表格
- Excel 导出：多 sheet .xlsx 文件，方便筛选排序

公共数据加载函数供 evaluator 复用。
"""

import json
import webbrowser
from pathlib import Path

import pandas as pd
from jinja2 import Environment, PackageLoader

from src.constants import RESULTS_DIR


def load_responses(path: Path) -> list[dict]:
    """从 responses.jsonl 加载结果，按 row_index 去重（保留最后一条）。"""
    responses_by_idx: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                responses_by_idx[entry["row_index"]] = entry
    return [responses_by_idx[i] for i in sorted(responses_by_idx)]


def load_scores(path: Path) -> dict[int, dict]:
    """从 scores.jsonl 加载评测分数。"""
    scores: dict[int, dict] = {}
    if not path.exists():
        return scores
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                scores[entry["row_index"]] = entry
    return scores


def _extract_response_text(response: dict) -> str:
    """从 API 响应中提取回复文本。"""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_reasoning_text(response: dict) -> str:
    """从 API 响应中提取推理文本（如有）。"""
    try:
        return response["choices"][0]["message"].get("reasoning_content", "")
    except (KeyError, IndexError, TypeError):
        return ""


def _enrich_analysis(analysis: str, scores: dict, dim_name_map: dict, dim_name_reverse: dict) -> str:
    """为评分分析文本中的每个维度名后注入分数。

    将 "相关性：完全切题..." 转换为 "相关性（5分）：完全切题..."
    如果分析文本中已经包含分数（如 "相关性（5分）："），则保持不变。

    Args:
        analysis: judge 输出的原始分析文本
        scores: 该行对应的评分字典，如 {"relevance": 5, "factuality": 4, ...}
        dim_name_map: 英文维度名到中文的映射
        dim_name_reverse: 中文维度名到英文的映射
    """
    if not analysis:
        return analysis

    import re

    # 构建中文维度名 → 英文维度名的反向映射
    reverse_map = {v: k for k, v in dim_name_map.items()}

    def replace_dim(match):
        chinese_name = match.group(1)
        english_name = reverse_map.get(chinese_name)
        if english_name and english_name in scores:
            return f"{chinese_name}（{scores[english_name]}分）："
        return match.group(0)

    # 匹配模式：维度中文名 + 可选的（N分）+ 冒号
    pattern = r"(相关性|事实性|流畅性|结构化|实时性|本地化|搜索质量|综合)(?:（\d+分）)?："
    return re.sub(pattern, replace_dim, analysis)


def _extract_usage(response: dict) -> dict:
    """从 API 响应中提取 token 用量。"""
    try:
        return response.get("usage") or {}
    except (KeyError, TypeError):
        return {}


def _extract_search_results(candidate_input: list[dict] | None) -> str:
    """从 candidate_input 中提取搜索结果（tool_calls 和 tool 消息）。

    Returns:
        格式化的搜索结果文本，如无搜索则返回空字符串
    """
    if not candidate_input:
        return ""

    parts = []
    for msg in candidate_input:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "")
                parts.append(f"[搜索关键词]\n{args}")
        elif role == "tool":
            content = msg.get("content", "")
            parts.append(f"[搜索结果]\n{content}")

    return "\n\n".join(parts)


def generate_html_report(run_name: str, open_browser: bool = False) -> Path:
    """生成 HTML 报告。

    Args:
        run_name: 实验运行名称
        open_browser: 是否自动打开浏览器

    Returns:
        生成的 HTML 文件路径
    """
    result_dir = RESULTS_DIR / run_name
    responses_path = result_dir / "responses.jsonl"
    scores_path = result_dir / "scores.jsonl"
    summary_path = result_dir / "summary.json"

    if not responses_path.exists():
        raise FileNotFoundError(f"No results for '{run_name}' at {responses_path}")

    responses = load_responses(responses_path)
    scores = load_scores(scores_path)

    # 读取 summary（如有）
    summary = None
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    # 维度名称映射（英文到中文）
    dim_name_map = {
        "relevance": "相关性",
        "factuality": "事实性",
        "fluency": "流畅性",
        "structure": "结构化",
        "timeliness": "实时性",
        "localization": "本地化",
        "search_quality": "搜索质量",
        "overall": "综合"
    }
    dim_name_reverse = {v: k for k, v in dim_name_map.items()}

    # 准备数据
    rows_data = []
    for r in responses:
        row_idx = r["row_index"]
        response = r.get("response", {})
        usage = _extract_usage(response)
        score = scores.get(row_idx, {})

        rows_data.append({
            "row_index": row_idx,
            "query": r.get("query", ""),
            "response": _extract_response_text(response),
            "reasoning": _extract_reasoning_text(response),
            "search_results": _extract_search_results(r.get("candidate_input")),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": r.get("latency_ms"),
            "ttft_ms": r.get("ttft_ms"),
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            "scores": {k: v for k, v in score.items() if k not in ("row_index", "query", "response_summary", "error", "analysis")},
            "analysis": _enrich_analysis(
                score.get("analysis", ""),
                {k: v for k, v in score.items() if k not in ("row_index", "query", "response_summary", "error", "analysis")},
                dim_name_map,
                dim_name_reverse,
            ),
        })

    # 统计数据
    latencies = [r["latency_ms"] for r in rows_data if r["latency_ms"] is not None]
    ttfts = [r["ttft_ms"] for r in rows_data if r["ttft_ms"] is not None]
    total_prompt_tokens = sum(r["prompt_tokens"] or 0 for r in rows_data)
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in rows_data)

    stats = {
        "total_rows": len(rows_data),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
    }

    # 评分汇总
    score_summary = summary.get("summary", {}) if summary else {}
    dimensions = summary.get("dimensions", []) if summary else []

    # 生成维度标签（中文名 + 平均分）
    dim_labels = {}
    for dim in dimensions:
        chinese_name = dim_name_map.get(dim, dim)
        avg_score = score_summary.get(f"avg_{dim}")
        if avg_score is not None:
            dim_labels[dim] = f"{chinese_name}（{avg_score}分）"
        else:
            dim_labels[dim] = chinese_name

    # 渲染 HTML
    env = Environment(loader=PackageLoader("src", "templates"))
    template = env.get_template("report.html")
    html = template.render(
        run_name=run_name,
        stats=stats,
        rows=rows_data,
        score_summary=score_summary,
        dimensions=dimensions,
        dim_labels=dim_labels,
        has_scores=bool(scores),
        has_ttft=bool(ttfts),
    )

    # 写入文件
    report_path = result_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        webbrowser.open(report_path.absolute().as_uri())

    return report_path


def export_excel(run_name: str) -> Path:
    """导出 Excel 文件。

    Args:
        run_name: 实验运行名称

    Returns:
        生成的 Excel 文件路径
    """
    result_dir = RESULTS_DIR / run_name
    responses_path = result_dir / "responses.jsonl"
    scores_path = result_dir / "scores.jsonl"
    summary_path = result_dir / "summary.json"

    if not responses_path.exists():
        raise FileNotFoundError(f"No results for '{run_name}' at {responses_path}")

    responses = load_responses(responses_path)
    scores = load_scores(scores_path)

    # Summary sheet
    summary_data = []
    summary_data.append({"Key": "Experiment", "Value": run_name})
    summary_data.append({"Key": "Total Rows", "Value": len(responses)})

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        summary_data.append({"Key": "Judge Model", "Value": summary.get("judge_model", "N/A")})
        for k, v in summary.get("summary", {}).items():
            summary_data.append({"Key": k, "Value": v})

    df_summary = pd.DataFrame(summary_data)

    # Responses sheet
    rows_data = []
    for r in responses:
        response = r.get("response", {})
        usage = _extract_usage(response)
        rows_data.append({
            "row_index": r["row_index"],
            "query": r.get("query", ""),
            "response": _extract_response_text(response),
            "reasoning_content": _extract_reasoning_text(response),
            "search_results": _extract_search_results(r.get("candidate_input")),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": r.get("latency_ms"),
            "ttft_ms": r.get("ttft_ms"),
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        })
    df_responses = pd.DataFrame(rows_data)

    # Scores sheet（如有）
    df_scores = None
    if scores:
        scores_data = []
        for row_idx, score in sorted(scores.items()):
            row = {"row_index": row_idx, "query": score.get("query", "")}
            for k, v in score.items():
                if k not in ("row_index", "query", "response_summary", "error"):
                    row[k] = v
            scores_data.append(row)
        df_scores = pd.DataFrame(scores_data)

    # 写入 Excel
    export_path = result_dir / "report.xlsx"
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_responses.to_excel(writer, sheet_name="Responses", index=False)
        if df_scores is not None:
            df_scores.to_excel(writer, sheet_name="Scores", index=False)

    return export_path
