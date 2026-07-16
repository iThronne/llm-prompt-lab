"""数据导入模块。

将已有的 Excel / JSONL 数据（Query + 模型回答）转换为 responses.jsonl 格式，
以便复用现有的 eval 流程进行 LLM-as-Judge 评测。

JSONL 源可承载超长 api_json（不受 Excel 单元格 32767 字符限制）。
"""

import json
from pathlib import Path

from src.config import ExperimentConfigLoader
from src.constants import RESULTS_DIR, META_FILE
from src.dataset import copy_dataset, _read_records, _normalize_api_json, _is_blank


def import_data(
        data_path: str,
        run_name: str,
        query_col: str = "query",
        response_col: str = "response",
        api_json_col: str = "api_json",
):
    """从 Excel / JSONL / CSV 导入数据，生成可供 eval 使用的 run 目录。

    Args:
        data_path: 数据文件路径（.xlsx / .jsonl / .csv）
        run_name: 生成的 run 名称
        query_col: Query 列名，默认 "query"
        response_col: 模型回答列名，默认 "response"
        api_json_col: api_json 列名，默认 "api_json"
    """

    # 读取数据（按扩展名分派）
    records = _read_records(Path(data_path))
    if not records:
        raise ValueError(f"数据文件 '{data_path}' 为空或无可读记录")

    for col in (query_col, response_col, api_json_col):
        if col not in records[0]:
            raise ValueError(
                f"Data file '{data_path}' must have a '{col}' column. "
                f"Found columns: {list(records[0].keys())}"
            )

    # api_json 归一化为字符串（JSONL 可能写成对象/数组）
    _normalize_api_json(records, api_json_col)

    # 创建 run 目录
    result_dir = RESULTS_DIR / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存数据集到 run 目录（仅保留用到的列，保证可追溯；保留源格式）
    cols_to_keep = [query_col, response_col, api_json_col]
    saved_dataset_path = copy_dataset(data_path, result_dir, cols_to_keep)

    # 写入 responses.jsonl
    responses_path = result_dir / "responses.jsonl"
    error_count = 0
    with open(responses_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(records):
            query_text = str(row.get(query_col, ""))
            response_text = str(row.get(response_col, ""))

            # 解析 api_json，构造 rendered_request
            api_json_raw = row.get(api_json_col)
            rendered_request = {"messages": []}
            if _is_blank(api_json_raw) or not str(api_json_raw).strip():
                pass  # 空值，保留空 messages
            else:
                try:
                    parsed = json.loads(str(api_json_raw))
                    if isinstance(parsed, list):
                        rendered_request = {"messages": parsed}
                    elif isinstance(parsed, dict) and "messages" in parsed:
                        rendered_request = parsed
                    else:
                        rendered_request = {
                            "error": f"api_json must be a messages array or an object "
                                     f"with 'messages' key, got: {type(parsed).__name__}"
                        }
                        error_count += 1
                except json.JSONDecodeError as e:
                    rendered_request = {"error": f"api_json parse error: {e}"}
                    error_count += 1

            record = {
                "experiment": run_name,
                "row_index": idx,
                "query": query_text,
                "language": row.get("language"),
                "location": row.get("location"),
                "rendered_request": rendered_request,
                "response": {
                    "choices": [{"message": {"content": response_text}}]
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if error_count > 0:
        print(f"[warn] {error_count} rows had invalid api_json")

    # 写入 meta.json
    # 导入的数据并非由当前配置的 candidate 模型产生，故不记录 candidate/prompt 信息
    meta = {
        "run_name": run_name,
        "source": "imported",
        "dataset": str(saved_dataset_path),
        "dataset_content_hash": ExperimentConfigLoader.hash_file(saved_dataset_path),
    }
    meta_path = result_dir / META_FILE
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] imported {len(records)} rows -> {result_dir}")
    print(f"  Run 'python -m src.cli eval {run_name}' to evaluate.")
