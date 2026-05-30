"""数据导入模块。

将已有的 Excel 数据（Query + 模型回答）转换为 responses.jsonl 格式，
以便复用现有的 eval 流程进行 LLM-as-Judge 评测。
"""

import json

import pandas as pd

from src.config import Config
from src.constants import RESULTS_DIR, META_FILE


def import_excel(
        excel_path: str,
        run_name: str,
        config: Config,
        query_col: str = "query",
        response_col: str = "response",
):
    """从 Excel 导入数据，生成可供 eval 使用的 run 目录。

    Args:
        excel_path: Excel 文件路径
        run_name: 生成的 run 名称
        config: 配置对象（用于获取 judge 配置）
        query_col: Query 列名，默认 "query"
        response_col: 模型回答列名，默认 "response"
    """
    exp = config.get_experiment()

    # 读取 Excel
    df = pd.read_excel(excel_path)
    for col in (query_col, response_col):
        if col not in df.columns:
            raise ValueError(
                f"Excel file '{excel_path}' must have a '{col}' column. "
                f"Found columns: {list(df.columns)}"
            )

    # 创建 run 目录
    result_dir = RESULTS_DIR / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 写入 responses.jsonl
    responses_path = result_dir / "responses.jsonl"
    with open(responses_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            response_text = str(row[response_col])
            record = {
                "experiment": run_name,
                "row_index": idx,
                "query": str(row[query_col]),
                "response": {
                    "choices": [{"message": {"content": response_text}}]
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 写入 meta.json
    # 注意：导入的数据并非由当前配置的 candidate 模型产生，故不记录 candidate 信息
    meta = {
        "run_name": run_name,
        "source": "imported",
        "prompt_name": exp.prompt_name,
        "prompt_content": exp.prompt,
        "dataset": excel_path,
        "judge": exp.judge.model_dump() if exp.judge else None,
    }
    meta_path = result_dir / META_FILE
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] imported {len(df)} rows → {result_dir}")
    print(f"  Run 'python -m src.cli eval {run_name}' to evaluate.")
