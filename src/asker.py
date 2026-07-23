"""单 case 追问模块。

针对某个已评测的 case，把"评分标准（judge prompt）+ 该 case 完整上下文"
喂给模型，回答用户的追问；每轮追问持久化到 results/<run>/qa.jsonl，
下次 report 时在对应 case 下方展示。

设计原则：最大化复用 advise 的"读 run 产物 -> 构造 messages -> call_model
-> 写产物"模式与 reporter 的 load/extract 工具，不重写读取与调用逻辑。
模型与 API key 复用 advise 配置（自带 enable_search，便于核实事实）。
"""

import json
from pathlib import Path

from src.advisor import _extract_rendered_system
from src.config import AdviseConfig
from src.constants import RESULTS_DIR
from src.models import call_model, create_client
from src.reporter import (
    _extract_response_text,
    _extract_reasoning_text,
    _extract_search_queries,
    _extract_search_results,
    load_qa,
    load_responses,
    load_scores,
)


ASK_SYSTEM_PROMPT = """你是一位严谨的 AI 回复评测复盘助手。用户会提供一次评测中某个 case 的完整上下文：评分标准（Judge Prompt）、候选模型实际收到的 System Prompt、用户 Query、候选模型回复、（如有）推理过程 / 搜索词 / 搜索结果、各维度分数与评测分析，以及此前就该 case 的多轮追问记录。

你的任务是：基于上述上下文，回答用户对该 case 的追问。常见追问包括：
- 解释某个维度为何得此分数、判定依据是什么
- 核实评测分析中引用的原文是否确实存在问题（如某条“事实错误”是否真的错误，必要时可联网验证）
- 评判评分是否合理、是否应当上调或下调，并给出理由
- 针对候选回复或 System Prompt 提出改进建议

要求：
- 回答须紧扣该 case 的具体内容，不要泛泛而谈
- 引用候选回复或搜索结果中的原文时，保持原文措辞不改写（用「」包裹），便于检索
- 优先基于评测分析中已引用的原文展开
- 若上下文不足以判断，如实说明，不要编造
- 输出自由 markdown
"""


def _build_case_context(row: dict, score: dict, judge_prompt: str) -> str:
    """拼装单个 case 的完整上下文（markdown），作为 user 消息。"""
    response = row.get("response", {})
    rendered_request = row.get("rendered_request")
    score_dims = {
        k: v for k, v in score.items()
        if k not in ("row_index", "query", "response_summary", "error", "analysis")
    }

    sections = [
        "## 评分标准（Judge Prompt）\n" + (judge_prompt or "（无）"),
        f"## 该 case 的完整上下文（row_index={row['row_index']}）",
        "### 候选模型实际收到的 System Prompt\n"
        + (_extract_rendered_system(rendered_request) or "（无）"),
        "### 用户 Query\n" + (row.get("query", "") or "（无）"),
        "### 候选模型回复\n" + (_extract_response_text(response) or "（无）"),
    ]

    reasoning = _extract_reasoning_text(response)
    if reasoning:
        sections.append("### 候选模型推理\n" + reasoning)

    search_queries = _extract_search_queries(rendered_request)
    if search_queries:
        sections.append("### 搜索关键词\n" + search_queries)

    search_results = _extract_search_results(rendered_request)
    if search_results:
        sections.append("### 搜索结果\n" + search_results)

    if score_dims:
        score_line = ", ".join(f"{k}={v}" for k, v in score_dims.items())
        sections.append("### 各维度分数\n" + score_line)

    analysis = score.get("analysis", "")
    if analysis:
        sections.append("### 评测分析\n" + analysis)

    return "\n\n".join(sections)


def _load_case_context(
    run_name: str, row_index: int, judge_prompt: str
) -> tuple[list[dict], Path, int]:
    """加载 case 并构造初始 messages（system + 上下文 + 历史 Q&A）。

    Returns:
        (messages, qa_path, prior_count)：messages 已含 system/上下文及历史追问；
        qa_path 为该 run 的 qa.jsonl；prior_count 为已存在的追问轮数。
    """
    result_dir = RESULTS_DIR / run_name
    responses_path = result_dir / "responses.jsonl"
    if not responses_path.exists():
        raise FileNotFoundError(f"No results for '{run_name}' at {responses_path}")

    responses = load_responses(responses_path)
    scores = load_scores(result_dir / "scores.jsonl")
    qa_path = result_dir / "qa.jsonl"

    row = next((r for r in responses if r["row_index"] == row_index), None)
    if row is None:
        raise FileNotFoundError(
            f"row_index={row_index} not found in {responses_path}"
        )

    score = scores.get(row_index, {})
    context = _build_case_context(row, score, judge_prompt)
    prior_qa = load_qa(qa_path).get(row_index, [])

    messages: list[dict] = [
        {"role": "system", "content": ASK_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    for t in prior_qa:
        messages.append({"role": "user", "content": t.get("question", "")})
        messages.append({"role": "assistant", "content": t.get("answer", "")})

    return messages, qa_path, len(prior_qa)


def append_qa(qa_path: Path, row_index: int, question: str, answer: str) -> int:
    """向 qa.jsonl 追加一条追问记录，返回该轮 turn 编号（从 1 递增）。"""
    existing = load_qa(qa_path).get(row_index, [])
    turn = len(existing) + 1
    entry = {"row_index": row_index, "turn": turn, "question": question, "answer": answer}
    with open(qa_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return turn


async def run_ask(
    run_name: str, row_index: int, question: str,
    advise_cfg: AdviseConfig, judge_prompt: str,
) -> str:
    """单次追问：加载 case 上下文 -> 调用模型 -> 追加 qa.jsonl -> 返回答案。"""
    messages, qa_path, _ = _load_case_context(run_name, row_index, judge_prompt)
    messages.append({"role": "user", "content": question})

    client = create_client(advise_cfg.model)
    response, _ = await call_model(client, advise_cfg.model, messages)
    answer = _extract_response_text(response)

    append_qa(qa_path, row_index, question, answer)
    return answer


async def run_ask_interactive(
    run_name: str, row_index: int,
    advise_cfg: AdviseConfig, judge_prompt: str,
) -> None:
    """交互式多轮追问：REPL 循环，每轮调用模型并追加 qa.jsonl，空行或 Ctrl-C 退出。"""
    messages, qa_path, prior_count = _load_case_context(run_name, row_index, judge_prompt)
    client = create_client(advise_cfg.model)

    print(f"[ask] run={run_name} row={row_index} 交互式追问（空行或 Ctrl-C 退出）")
    if prior_count:
        print(f"[ask] 已加载 {prior_count} 轮历史追问作为上下文")

    while True:
        try:
            question = input("\n你的追问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ask] 退出")
            break
        if not question:
            break

        messages.append({"role": "user", "content": question})
        try:
            response, _ = await call_model(client, advise_cfg.model, messages)
        except Exception as e:
            print(f"[error] 调用失败: {e}")
            messages.pop()  # 回滚未成功的提问，保持历史一致
            continue

        answer = _extract_response_text(response)
        messages.append({"role": "assistant", "content": answer})
        turn = append_qa(qa_path, row_index, question, answer)
        print(f"\n[A{turn}] {answer}")
