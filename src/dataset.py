"""数据集加载与模板替换模块。

从 Excel 读取原始数据集，运行时用 Jinja2 将占位符替换为实际值。
Excel 约定列名：query (用户问题), api_json (完整 API 调用 JSON 字符串)。
api_json 中的任意字段都可以使用 {{ var_name }} Jinja2 占位符。
"""

import json
from pathlib import Path

import pandas as pd
from jinja2 import Template

# 数据集必需列
REQUIRED_COLUMNS = ["query", "api_json"]
# 数据集可选列（用于 locale 上下文注入）
OPTIONAL_COLUMNS = ["language", "location"]


def copy_dataset(src_path: str, dest_path: Path, columns: list[str]) -> Path:
    """复制数据集到指定路径，仅保留指定的列。

    Args:
        src_path: 源 Excel 文件路径
        dest_path: 目标文件路径（目录或完整路径）
        columns: 要保留的列名列表

    Returns:
        实际写入的文件路径
    """
    src = Path(src_path)
    dest = Path(dest_path)

    # 如果 dest_path 是目录，使用源文件名
    if dest.is_dir() or not dest.suffix:
        dest = dest / src.name

    dest.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(src_path)
    # 只保留存在的列
    cols_to_keep = [c for c in columns if c in df.columns]
    df[cols_to_keep].to_excel(dest, index=False)

    return dest


def load_dataset(excel_path: str) -> list[dict]:
    """从 Excel 加载数据集，返回 list[dict]。

    必选列：query, api_json
    可选列：language, location（用于 locale 上下文注入）
    """
    df = pd.read_excel(excel_path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Excel file '{excel_path}' must have a '{col}' column")

    cols = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    records = df[cols].to_dict("records")
    # 填充空值（NaN → None）
    for r in records:
        for key in ("language", "location"):
            if key not in r or pd.isna(r.get(key)):
                r[key] = None
    return records


def render_request(api_json_str: str, variables: dict) -> dict:
    """用 Jinja2 渲染 api_json 模板字符串，返回解析后的 dict。

    Args:
        api_json_str: 包含 {{ placeholder }} 的 JSON 模板字符串
        variables: 模板变量映射（system_prompt, query 等）

    Returns:
        渲染并解析后的 API 请求 dict
    """
    template = Template(api_json_str)
    rendered = template.render(**variables)
    return json.loads(rendered)
