"""数据集加载与模板替换模块。

从 Excel 读取原始数据集，运行时用 Jinja2 将占位符替换为实际值。
Excel 约定列名：query (用户问题), api_json (完整 API 调用 JSON 字符串)。
api_json 中的任意字段都可以使用 {{ var_name }} Jinja2 占位符。
"""

import json

import pandas as pd
from jinja2 import Template


def load_dataset(excel_path: str) -> list[dict]:
    """从 Excel 加载数据集，返回 list[dict]。

    必选列：query, api_json
    可选列：language, location（用于 locale 上下文注入）
    """
    df = pd.read_excel(excel_path)
    if "query" not in df.columns:
        raise ValueError(f"Excel file '{excel_path}' must have a 'query' column")
    if "api_json" not in df.columns:
        raise ValueError(f"Excel file '{excel_path}' must have an 'api_json' column")

    # 基础列
    cols = ["query", "api_json"]
    # 可选 locale 列
    for col in ("language", "location"):
        if col in df.columns:
            cols.append(col)

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
