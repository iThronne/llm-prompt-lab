"""独立归因执行器：仅读取候选输入，不读取任何 Judge 产物。"""

import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from src.attribution_schema import AttributionResult, PIPELINE_VERSION, SCHEMA_VERSION, parse_result
from src.config import AttributionConfig
from src.constants import RESULTS_DIR
from src.models import create_client, call_model, call_model_stream
from src.sanitizer import compile_rules, sanitize_text


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def offline_model(config: AttributionConfig):
    """拒绝显式工具/联网配置。未启用的供应商选项不强行传给其他 API。"""
    params = config.model.model_dump()

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"tools", "functions", "enable_search", "web_search", "search_options",
                           "web_search_options", "search_parameters"} and item:
                    raise ValueError(f"归因只允许提供的材料，禁止模型配置 {key}")
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(params)
    return config.model


def config_hash(config: AttributionConfig) -> str:
    semantic_config = config.model_dump(exclude={"concurrency", "retries", "retry_delay"})
    return digest({"config": semantic_config, "schema": AttributionResult.model_json_schema(),
                   "pipeline_version": PIPELINE_VERSION})[:16]


def read_responses(path: Path) -> list[dict]:
    """严格读取源数据，重复行号拒绝处理，避免悄悄覆盖案例。"""
    records = []
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("row_index")
            if type(index) is not int or index < 0 or index in seen:
                raise ValueError(f"{path}:{line_number} row_index 无效或重复")
            seen.add(index)
            records.append(row)
    return records


def _text(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def build_input(row: dict, config: AttributionConfig) -> tuple[dict, list[str], dict]:
    quality = []
    request = row.get("rendered_request")
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list):
        quality.append("messages 缺失或不是数组，无法确认执行日志完整性。")
        messages = []
    elif not messages:
        quality.append("messages 为空，无法根据空日志断言候选没有搜索。")
    if isinstance(request, dict) and request.get("error"):
        quality.append("源 rendered_request 记录了解析错误。")
    try:
        answer = row["response"]["choices"][0]["message"].get("content")
        if row["response"]["choices"][0].get("finish_reason") == "length":
            quality.append("候选响应 finish_reason=length，回答可能被截断。")
    except (KeyError, IndexError, TypeError, AttributeError):
        answer = row.get("answer")
        quality.append("原始 response 不完整，使用 answer 字段（若有）。")
    if not answer:
        quality.append("模型答复为空或缺失；不能直接推断为空回复的工程原因。")
    note = row.get("human_note")
    if note is None:
        note = row.get("note")
    snapshot = {"query": _text(row.get("query")) or "", "messages": messages,
                "answer": _text(answer) or "", "human_note": _text(note) or None}
    if not snapshot["query"]:
        quality.append("Query 缺失。")
    if snapshot["human_note"]:
        bound = row.get("human_note_answer_hash")
        if not bound:
            quality.append("人工评论未绑定具体回答版本；需检查其适用性，不默认评价当前回答。")
        elif bound != digest(snapshot["answer"]):
            quality.append("人工评论绑定的回答与当前回答不同，仅保留为参考，不能直接套用。")

    compiled = compile_rules(config.sanitize)

    def clean(value):
        if isinstance(value, str):
            return sanitize_text(value, compiled)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    snapshot = clean(snapshot)
    trace = tool_trace(snapshot["messages"], config.search_tool_names)
    quality.extend(trace.pop("warnings"))
    return snapshot, quality, trace


def tool_trace(messages: list, search_names: list[str]) -> dict:
    """按明确配置的名称标记搜索；未知工具保留原名、定位和关联。"""
    names = {name.casefold() for name in search_names}
    calls, results, warnings = [], [], []
    call_ids = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            warnings.append(f"messages[{index}] 不是对象。")
            continue
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            warnings.append(f"messages[{index}].tool_calls 不是数组。")
            raw_calls = []
        for offset, call in enumerate(raw_calls):
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                warnings.append(f"messages[{index}] 含无法解析的工具调用。")
                continue
            function = call["function"]
            name = str(function.get("name", ""))
            call_id = call.get("id")
            if not isinstance(call_id, str):
                call_id = None
            entry = {"path": f"/messages/{index}/tool_calls/{offset}",
                     "tool_call_id": call_id, "name": name,
                     "is_search": name.casefold() in names}
            calls.append(entry)
            if entry["tool_call_id"]:
                if entry["tool_call_id"] in call_ids:
                    warnings.append(f"工具调用 ID 重复: {entry['tool_call_id']}")
                call_ids[entry["tool_call_id"]] = entry
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str):
                call_id = None
            linked = call_ids.get(call_id)
            results.append({"path": f"/messages/{index}/content",
                            "tool_call_id": call_id,
                            "call_path": linked["path"] if linked else None,
                            "is_search": linked["is_search"] if linked else None})
            if linked is None:
                warnings.append(f"messages[{index}] 的工具结果未关联到先前调用。")
    returned = {r["tool_call_id"] for r in results if r["call_path"]}
    for call in calls:
        if not call["tool_call_id"] or call["tool_call_id"] not in returned:
            warnings.append(f"工具调用 {call['path']} 缺少可关联的返回结果。")
    return {"calls": calls, "results": results, "warnings": warnings}


def build_messages(config, snapshot, quality, trace):
    schema = json.dumps(AttributionResult.model_json_schema(), ensure_ascii=False)
    return [
        {"role": "system", "content": config.prompt + "\n\n输出必须满足以下 JSON Schema：\n" + schema},
        {"role": "user", "content": json.dumps({"input_snapshot": snapshot,
         "input_quality": quality, "tool_trace": trace}, ensure_ascii=False)},
    ]


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    raw = path.read_bytes()
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # 只容忍进程中断造成的最后一行不完整；内部损坏必须显式修复。
            if index != len(lines) - 1 or raw.endswith(b"\n"):
                raise ValueError(f"归因日志第 {index + 1} 行损坏: {path}")
            print(f"[warn] 忽略未写完的归因日志末行: {path}")
    return records


@contextmanager
def directory_lock(directory: Path):
    lock = directory / ".write.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"归因目录正在写入，或上次异常退出遗留锁：{lock}。确认无任务运行后可删除此锁。") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def append_record(path: Path, record):
    # 在锁保护下恢复末行：备份原文件，再去掉未完成的最后一条。
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            incomplete_tail = stream.read(1) != b"\n"
        if incomplete_tail:
            raw = path.read_bytes()
            tail = raw.rsplit(b"\n", 1)[-1]
            try:
                json.loads(tail)
                with path.open("ab") as stream:
                    stream.write(b"\n")
            except (json.JSONDecodeError, UnicodeDecodeError):
                backup = path.with_name(f"{path.name}.{utc_now().replace(':', '-')}.partial")
                backup.write_bytes(raw)
                with path.open("r+b") as stream:
                    stream.truncate(len(raw) - len(tail))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_cases(rows, config, run_name):
    prepared = []
    for row in rows:
        snapshot, quality, trace = build_input(row, config)
        prepared.append({"schema_version": SCHEMA_VERSION,
                         "case_id": f"{run_name}:{row['row_index']}", "row_index": row["row_index"],
                         "input_hash": digest({"input": snapshot, "quality": quality, "trace": trace}),
                         "input_snapshot": snapshot, "input_quality": quality, "tool_trace": trace})
    return prepared


def current_records(cases, journal, hash_value):
    latest = {(r["case_id"], r["input_hash"]): r for r in journal if r.get("config_hash") == hash_value}
    records = []
    for case in cases:
        record = latest.get((case["case_id"], case["input_hash"]))
        records.append(record or {**case, "config_hash": hash_value,
                                  "execution": {"status": "not_processed"}, "result": None})
    return records


def summarize(records):
    executions, statuses, primary, joint, involved, types, notes = (Counter() for _ in range(7))
    failures = Counter()
    for record in records:
        executions[record["execution"]["status"]] += 1
        if record["execution"]["status"] == "failed":
            failures[record["execution"].get("error_type", "unknown")] += 1
        result = record.get("result")
        if not result:
            continue
        statuses[result["status"]] += 1
        factors = result["factors"]
        selected = {f["stage"] for f in factors if f["id"] in result["primary_factor_ids"]}
        if result["status"] == "primary_identified":
            primary.update(selected)
        elif result["status"] == "joint_primary":
            joint.update(selected)
        involved.update({f["stage"] for f in factors})
        types.update({(f["stage"] + "/" + f["problem_type"]) for f in factors})
        notes.update(n["assessment"] for n in result["note_assessment"])
    return {"total_cases": len(records), "execution_counts": dict(executions),
            "status_counts": dict(statuses), "unique_primary_cases_by_stage": dict(primary),
            "joint_primary_cases_by_stage": dict(joint), "involved_cases_by_stage": dict(involved),
            "problem_type_case_counts": dict(types), "note_claim_counts": dict(notes),
            "failure_counts": dict(failures),
            "counting_note": "环节/类型按案例去重；评论按主张计数。涉及多个环节的案例会重复计入，不能相加作为责任占比。"}


async def run_attribution(run_name: str, config: AttributionConfig, *, rows: list[int] | None = None,
                          force: bool = False, report_only: bool = False,
                          formats: tuple[str, ...] = ("html", "xlsx")) -> Path:
    model = offline_model(config)
    all_rows = read_responses(RESULTS_DIR / run_name / "responses.jsonl")
    if rows is not None and not set(rows) <= {r["row_index"] for r in all_rows}:
        raise ValueError("--rows 包含源数据中不存在的 row_index")
    cases = prepare_cases(all_rows, config, run_name)
    hash_value = config_hash(config)
    directory = RESULTS_DIR / run_name / "attribution" / hash_value
    if report_only and not (directory / "attributions.jsonl").exists():
        raise FileNotFoundError(f"当前配置没有归因结果: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "attributions.jsonl"
    with directory_lock(directory):
        journal = load_journal(journal_path)
        current = current_records(cases, journal, hash_value)
        pending = [r for r in current if (rows is None or r["row_index"] in rows)
                   and (force or r["execution"]["status"] != "success")]
        if not report_only:
            write_json(directory / "attribution_meta.json", {
                "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                "config_hash": hash_value, "config": config.model_dump(),
                "updated_at": utc_now(), "run_name": run_name,
                "source": "../../responses.jsonl", "output_schema": AttributionResult.model_json_schema(),
            })
            semaphore = asyncio.Semaphore(config.concurrency)
            client = None

            async def analyze(record):
                nonlocal client
                async with semaphore:
                    entry = {k: v for k, v in record.items() if k not in ("execution", "result")}
                    messages = build_messages(config, entry["input_snapshot"], entry["input_quality"], entry["tool_trace"])
                    entry["execution"] = {"status": "failed", "started_at": utc_now(), "attempts": 0}
                    entry["result"] = None
                    if len(json.dumps(messages, ensure_ascii=False)) > config.max_input_chars:
                        entry["execution"]["error_type"] = "input_too_large"
                        entry["execution"]["error"] = "input_too_large: 输入超出 max_input_chars，未截断、未调用模型。"
                    else:
                        for attempt in range(config.retries + 1):
                            entry["execution"]["attempts"] = attempt + 1
                            try:
                                if client is None:
                                    client = create_client(model)
                                if model.stream:
                                    response, _, _ = await call_model_stream(client, model, messages)
                                else:
                                    response, _ = await call_model(client, model, messages)
                                choice = response["choices"][0]
                                if choice.get("finish_reason") == "length":
                                    raise ValueError("归因输出被截断，请增加 max_tokens")
                                if choice["message"].get("tool_calls"):
                                    raise ValueError("归因不允许发起工具调用")
                                entry["result"] = parse_result(choice["message"]["content"], entry["input_snapshot"])
                                entry["execution"]["status"] = "success"
                                entry["execution"].pop("error", None)
                                entry["execution"].pop("error_type", None)
                                break
                            except Exception as exc:
                                entry["execution"]["error_type"] = type(exc).__name__
                                # 不持久化供应商错误原文，以免其中回显未脱敏内容或凭据。
                                entry["execution"]["error"] = f"{type(exc).__name__}: 调用或结构/证据校验失败。"
                                if isinstance(exc, ValueError) and type(exc).__name__ != "ValidationError":
                                    entry["execution"]["error"] = str(exc)[:1000]
                                if type(exc).__name__ == "ValidationError":
                                    errors = exc.errors(include_input=False, include_url=False)
                                    entry["execution"]["error"] = json.dumps(errors, ensure_ascii=False)[:1500]
                                if isinstance(exc, ValueError):
                                    messages = messages[:2] + [{"role": "user", "content":
                                        "上次输出未通过结构/证据校验，请重新输出完整 JSON，并修正："
                                        + entry["execution"]["error"]}]
                                if attempt < config.retries:
                                    await asyncio.sleep(min(config.retry_delay * 2 ** attempt, 30))
                    entry["execution"]["finished_at"] = utc_now()
                    append_record(journal_path, entry)
                    journal.append(entry)
                    print(f"[attribute] row {entry['row_index']}: {entry['execution']['status']}")

            print(f"[attribute] 待处理 {len(pending)}/{len(cases)} 条，配置 {hash_value}")
            try:
                await asyncio.gather(*(analyze(record) for record in pending))
            finally:
                if client is not None:
                    await client.close()
            if not journal_path.exists():
                journal_path.touch()
        records = current_records(cases, journal, hash_value)
        summary = summarize(records)
        write_json(directory / "attribution_summary.json", summary)
        from src.attribution_report import generate_reports
        generate_reports(directory, records, summary, formats)
        print(f"[done] 归因结果 → {directory}; {summary['execution_counts']}")
    return directory
