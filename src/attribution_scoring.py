"""归因报告的独立评分：复用 eval 量表、请求与解析，独立保存版本化结果。"""

import asyncio
import json

from src.attribution import append_record, directory_lock, load_journal, utc_now, write_json
from src.config import EvalConfigLoader
from src.evaluator import (compute_eval_hash, score_input_hash, build_score_messages,
                           _is_empty_response, _parse_judge_output, MAX_RETRIES, RETRY_BASE_DELAY)
from src.models import create_client, call_model, call_model_stream


async def score_cases(run_dir, rows, selected, *, concurrency=1, report_only=False, config=None):
    cfg = config if config is not None else EvalConfigLoader().get_eval()
    if "overall" not in cfg.dimensions:
        raise ValueError("评分配置必须包含 overall 指标；不会自行平均分项生成总分")
    config_hash = compute_eval_hash(cfg)
    directory = run_dir / "scoring" / config_hash
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scores.jsonl"
    with directory_lock(directory):
        journal = load_journal(path)
        latest = {(r["row_index"], r["score_input_hash"]): r for r in journal}
        # 旧 eval 结果只有同时携带输入和配置版本时才可复用。
        legacy = run_dir / "scores.jsonl"
        reusable = {}
        if legacy.exists():
            for line in legacy.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("eval_config_hash") == config_hash and item.get("score_input_hash") and not item.get("error"):
                    try:
                        parsed = _parse_judge_output(json.dumps(item), cfg.dimensions)
                    except (ValueError, TypeError):
                        continue
                    reusable[(item["row_index"], item["score_input_hash"])] = parsed
        current = {}
        pending = []
        for row in rows:
            index = row["row_index"]
            key = (index, score_input_hash(row))
            record = latest.get(key)
            if not record or record["status"] not in ("success", "unscorable"):
                record = {"row_index": index, "score_input_hash": key[1], "eval_config_hash": config_hash,
                          "status": "not_processed", "result": None, "overall": None,
                          "dimensions": cfg.dimensions, "locator": f"../../scoring/{config_hash}/scores.jsonl"}
                if key in reusable:
                    record.update(status="success" if reusable[key].get("overall") is not None else "unscorable",
                                  result=reusable[key], overall=reusable[key].get("overall"), source="eval",
                                  judge_messages=build_score_messages(row, cfg))
                    if not report_only:
                        append_record(path, record)
                elif key in latest:
                    record = latest[key]
            current[index] = record
            if index in selected and record["status"] not in ("success", "unscorable"):
                pending.append((row, record))
        if report_only:
            return current
        write_json(directory / "score_meta.json", {"eval_config_hash": config_hash, "config": cfg.model_dump()})
        semaphore = asyncio.Semaphore(concurrency)
        client = None

        async def evaluate(row, record):
            nonlocal client
            async with semaphore:
                record.update(status="failed", attempts=0, result=None, overall=None, started_at=utc_now())
                record.pop("error", None)
                if _is_empty_response(row.get("response")):
                    record.update(status="unscorable", error="候选回复为空，沿用 eval 规则跳过评分。")
                else:
                    for attempt in range(MAX_RETRIES + 1):
                        record["attempts"] = attempt + 1
                        try:
                            messages = build_score_messages(row, cfg)
                            record["judge_messages"] = messages
                            if client is None:
                                client = create_client(cfg.model)
                            if cfg.model.stream:
                                response, _, _ = await call_model_stream(client, cfg.model, messages)
                            else:
                                response, _ = await call_model(client, cfg.model, messages)
                            choice = response["choices"][0]
                            if choice.get("finish_reason") == "length" or choice["message"].get("tool_calls"):
                                raise ValueError("评分输出截断或未返回最终答案")
                            result = _parse_judge_output(choice["message"]["content"], cfg.dimensions)
                            record.update(result=result, overall=result["overall"],
                                          status="success" if result["overall"] is not None else "unscorable")
                            record.pop("error", None)
                            break
                        except Exception as exc:
                            record["error"] = f"{type(exc).__name__}: 评分调用或校验失败。"
                            if attempt < MAX_RETRIES:
                                await asyncio.sleep(RETRY_BASE_DELAY * 2 ** attempt)
                record["finished_at"] = utc_now()
                append_record(path, record)
                print(f"[score] row {record['row_index']}: {record['status']}")
        try:
            await asyncio.gather(*(evaluate(row, record) for row, record in pending))
        finally:
            if client is not None:
                await client.close()
        return current
