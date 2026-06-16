"""数据集加载与消息构建模块。

从 Excel 读取原始数据集，解析 api_json 并构建最终的 messages 列表。
Excel 约定列名：query (用户问题), api_json (消息结构 JSON 字符串)。
api_json 可以是 messages 数组或包含 messages 键的完整请求对象。
运行时框架会强制覆盖 system 消息的 content（来自 config/prompts 文件）。
"""

import copy
import json
from pathlib import Path

import pandas as pd

# 数据集必需列
REQUIRED_COLUMNS = ["query", "api_json"]
# 数据集可选列（用于 locale / 垂域 上下文注入）
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
    可选列：language, location（用于 locale 上下文注入）, domain（用于垂域评分）
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


def build_messages(api_json_str: str, system_prompt: str) -> list[dict]:
    """解析 api_json 并构建最终的 messages 列表。

    设计原则：
    - api_json 定义消息结构（roles、顺序、多轮历史）
    - system_prompt 强制覆盖 system 消息的 content（来自 config/prompts 文件）
    - 其他消息（user、assistant 等）保持不变

    Args:
        api_json_str: 消息结构 JSON 字符串，支持两种格式：
            - messages 数组：[{"role": "system", ...}, {"role": "user", ...}]
            - 完整请求对象：{"model": "...", "messages": [...]}
        system_prompt: 要注入的 system prompt 内容（来自 prompt 文件 + locale 后缀）

    Returns:
        处理后的 messages 列表
    """
    parsed = json.loads(api_json_str)

    # 归一化：兼容 list 和 dict 两种格式
    if isinstance(parsed, list):
        messages = copy.deepcopy(parsed)
    elif isinstance(parsed, dict) and "messages" in parsed:
        messages = copy.deepcopy(parsed["messages"])
    else:
        raise ValueError(
            f"api_json must be a messages array or an object with 'messages' key, "
            f"got: {type(parsed).__name__}"
        )

    # 强制覆盖 system 消息的 content
    has_system = False
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = system_prompt
            has_system = True
            break

    # 若无 system 消息，在头部插入
    if not has_system:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return messages
