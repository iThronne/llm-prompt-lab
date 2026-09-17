"""归因输出契约及证据校验，不依赖评分模块。"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "1.0"
Stage = Literal["query", "search_query", "search_result", "context", "answer"]
STAGES = {
    "query": "用户 Query", "search_query": "搜索关键词", "search_result": "搜索结果",
    "context": "其他上下文", "answer": "模型答复",
}
STATUSES = {
    "primary_identified": "已确定主因", "joint_primary": "并列主要因素",
    "ranking_unresolved": "主次未定", "insufficient_evidence": "证据不足",
    "no_supported_issue": "未发现材料支持的缺陷",
}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(Record):
    id: str = Field(min_length=1)
    source: Literal["query", "answer", "messages", "human_note"]
    path: str = Field(description="相对于 input_snapshot 的 JSON Pointer，必须定位到字符串")
    quote: str = Field(min_length=1)
    source_language: str = Field(min_length=1)
    translation_zh: str | None
    translation_note: str | None


class NoteAssessment(Record):
    claim: str = Field(min_length=1)
    assessment: Literal["supported", "refuted", "unverifiable"]
    reason: str = Field(min_length=1)
    evidence_ids: list[str]


class StageCheck(Record):
    stage: Stage
    status: Literal["checked", "not_applicable", "unavailable"]
    finding: str = Field(min_length=1)
    evidence_ids: list[str]


class Issue(Record):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    impact_level: Literal["core", "important", "local"]
    evidence_ids: list[str] = Field(min_length=1)


class Fix(Record):
    action: str = Field(min_length=1)
    validation: str = Field(min_length=1)


class Factor(Record):
    id: str = Field(min_length=1)
    stage: Stage
    problem_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    issue_ids: list[str] = Field(min_length=1)
    priority: int | None = Field(ge=1)
    impact_level: Literal["core", "important", "local"]
    evidence_strength: Literal["direct", "indirect"]
    evidence_ids: list[str] = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    ranking_reason: str = Field(min_length=1)
    fix: Fix


class Suggestion(Record):
    stage: Stage
    description: str = Field(min_length=1)
    evidence_ids: list[str]
    fix: Fix


class OpenQuestion(Record):
    question: str = Field(min_length=1)
    needed_material: str = Field(min_length=1)
    limits: str = Field(min_length=1)


class Summaries(Record):
    query: str | None
    answer: str | None
    human_note: str | None


class AttributionResult(Record):
    status: Literal["primary_identified", "joint_primary", "ranking_unresolved",
                    "insufficient_evidence", "no_supported_issue"]
    content_summaries_zh: Summaries
    note_assessment: list[NoteAssessment]
    stage_checks: list[StageCheck]
    issues: list[Issue]
    evidence: list[Evidence]
    factors: list[Factor]
    primary_factor_ids: list[str]
    suggestions: list[Suggestion]
    open_questions: list[OpenQuestion]
    summary: str = Field(min_length=1)


def resolve_pointer(snapshot: dict, pointer: str):
    """仅解析 JSON Pointer，不运行输入中的代码或指令。"""
    if not pointer.startswith("/"):
        raise ValueError(f"证据路径不是 JSON Pointer: {pointer}")
    value = snapshot
    try:
        for part in pointer[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not part.isdigit():
                    raise ValueError("数组下标无效")
                value = value[int(part)]
            elif isinstance(value, dict):
                value = value[part]
            else:
                raise ValueError("路径不是容器")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"证据路径不存在: {pointer}") from exc
    return value


def parse_result(content: str, snapshot: dict) -> dict:
    """验证结构、引用真实性、跨表关联和主次状态。"""
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = AttributionResult.model_validate(json.loads(content))

    def unique(items, name):
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} ID 重复")
        return set(ids)

    evidence_ids = unique(result.evidence, "evidence")
    issue_ids = unique(result.issues, "issues")
    factor_ids = unique(result.factors, "factors")
    for evidence in result.evidence:
        if evidence.path.split("/")[1:2] != [evidence.source]:
            raise ValueError(f"证据 source/path 不一致: {evidence.id}")
        original = resolve_pointer(snapshot, evidence.path)
        if not isinstance(original, str) or evidence.quote not in original:
            raise ValueError(f"证据 {evidence.id} 的 quote 不是路径处的连续原文")
        language = evidence.source_language.lower().split("-")[0]
        if language not in ("zh", "en") and not evidence.translation_zh:
            raise ValueError(f"非中英文证据 {evidence.id} 缺少中文释义")

    evidence_map = {e.id: e for e in result.evidence}
    for item in [*result.note_assessment, *result.stage_checks, *result.issues,
                 *result.factors, *result.suggestions]:
        if not set(item.evidence_ids) <= evidence_ids:
            raise ValueError("引用了不存在的 evidence ID")
    for item in [*result.issues, *result.factors]:
        if all(evidence_map[e].source == "human_note" for e in item.evidence_ids):
            raise ValueError("已确认缺陷/因素不能仅以人工评论为证据")
    if sorted(check.stage for check in result.stage_checks) != sorted(STAGES):
        raise ValueError("stage_checks 必须包含且仅包含五个环节")
    for factor in result.factors:
        if not set(factor.issue_ids) <= issue_ids:
            raise ValueError("因素引用了不存在的 issue ID")
    primaries = result.primary_factor_ids
    if len(set(primaries)) != len(primaries) or not set(primaries) <= factor_ids:
        raise ValueError("primary_factor_ids 重复或不存在")
    ranked_first = {f.id for f in result.factors if f.priority == 1}
    if result.status in ("primary_identified", "joint_primary"):
        expected = len(primaries) == 1 if result.status == "primary_identified" else len(primaries) >= 2
        if not expected or set(primaries) != ranked_first:
            raise ValueError("主要因素数量/priority 与 status 不一致")
    elif primaries or ranked_first:
        raise ValueError("未确定主因时 primary_factor_ids 必须为空且不得设 priority=1")
    if result.status == "ranking_unresolved" and len(result.factors) < 2:
        raise ValueError("主次未定必须已有多个因素")
    if result.status == "no_supported_issue" and (result.issues or result.factors):
        raise ValueError("no_supported_issue 不得包含已确认缺陷或因素")
    if result.status in ("ranking_unresolved", "insufficient_evidence") and not result.open_questions:
        raise ValueError("未决结果必须说明待确认事项")
    if snapshot.get("human_note") and not result.note_assessment:
        raise ValueError("有人工评论时必须核对评论")
    if not snapshot.get("human_note") and result.note_assessment:
        raise ValueError("没有人工评论时不得虚构评论核对")
    return result.model_dump()
