"""数据导入模块。

将已有的 Excel 数据（Query + 模型回答）转换为 responses.jsonl 格式，
以便复用现有的 eval 流程进行 LLM-as-Judge 评测。
"""

import json

import pandas as pd

from src.config import ExperimentConfigLoader
from src.constants import RESULTS_DIR, META_FILE
from src.dataset import copy_dataset


def import_excel(
        excel_path: str,
        run_name: str,
        config: ExperimentConfigLoader,
        query_col: str = "query",
        response_col: str = "response",
        api_json_col: str = "api_json",
        domain_col: str = "domain",
        profile_name: str = "",
):
    """从 Excel 导入数据，生成可供 eval 使用的 run 目录。

    Args:
        excel_path: Excel 文件路径
        run_name: 生成的 run 名称
        config: 配置对象（用于获取 prompt 等配置）
        query_col: Query 列名，默认 "query"
        response_col: 模型回答列名，默认 "response"
        api_json_col: api_json 列名，默认 "api_json"
        domain_col: 垂域分类列名，默认 "domain"（可选，不存在则忽略）
        profile_name: 使用的 profile 名称
    """
    exp = config.get_experiment()

    # 读取 Excel
    df = pd.read_excel(excel_path)
    for col in (query_col, response_col, api_json_col):
        if col not in df.columns:
            raise ValueError(
                f"Excel file '{excel_path}' must have a '{col}' column. "
                f"Found columns: {list(df.columns)}"
            )

    # 创建 run 目录
    result_dir = RESULTS_DIR / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存数据集到 run 目录（仅保留用到的列，保证可追溯）
    cols_to_keep = [query_col, response_col, api_json_col]
    if domain_col in df.columns:
        cols_to_keep.append(domain_col)
    saved_dataset_path = copy_dataset(excel_path, result_dir, cols_to_keep)

    # 写入 responses.jsonl
    responses_path = result_dir / "responses.jsonl"
    error_count = 0
    with open(responses_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            query_text = str(row[query_col])
            response_text = str(row[response_col])

            # 解析 api_json，构造 rendered_request
            api_json_raw = row[api_json_col]
            rendered_request = {"messages": []}
            if pd.isna(api_json_raw) or not str(api_json_raw).strip():
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
                "language": row.get("language") if "language" in df.columns else None,
                "location": row.get("location") if "location" in df.columns else None,
                "domain": row.get(domain_col) if domain_col in df.columns else None,
                "rendered_request": rendered_request,
                "response": {
                    "choices": [{"message": {"content": response_text}}]
                },
            }
            # NaN → None
            if record["domain"] is not None and pd.isna(record["domain"]):
                record["domain"] = None
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if error_count > 0:
        print(f"[warn] {error_count} rows had invalid api_json")

    # 写入 meta.json
    # 注意：导入的数据并非由当前配置的 candidate 模型产生，故不记录 candidate 信息
    meta = {
        "run_name": run_name,
        "profile": profile_name,
        "source": "imported",
        "prompt_name": exp.prompt_name,
        "prompt_content": exp.prompt,
        "dataset": str(saved_dataset_path),
        "dataset_content_hash": ExperimentConfigLoader.hash_file(saved_dataset_path),
    }
    meta_path = result_dir / META_FILE
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] imported {len(df)} rows → {result_dir}")
    print(f"  Run 'python -m src.cli eval {run_name}' to evaluate.")
