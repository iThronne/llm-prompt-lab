"""数据集加载与消息构建模块。

从 Excel / JSONL 读取原始数据集，解析 api_json 并构建最终的 messages 列表。
数据集约定列名：query (用户问题), api_json (消息结构 JSON 字符串)。
api_json 可以是 messages 数组或包含 messages 键的完整请求对象。
运行时框架会强制覆盖 system 消息的 content（来自 config/prompts 文件）。

支持的数据集文件格式（按扩展名分派）：
  - .jsonl  每行一条 JSON 记录，字段无长度上限（推荐用于超长 api_json）
  - .xlsx/.xls  Excel 表格（单元格上限 32767 字符）
  - .csv   逗号分隔
"""

import copy
import json
from pathlib import Path

import pandas as pd

# 数据集必需列
REQUIRED_COLUMNS = ["query", "api_json"]
# 数据集可选列（用于 locale / 垂域 上下文注入）
OPTIONAL_COLUMNS = ["language", "location"]


def _read_records(path: Path) -> list[dict]:
    """按扩展名读取数据集，返回 list[dict]。

    - .jsonl：逐行 json.loads，字段无长度上限（api_json 可超 32767）
    - .csv：pd.read_csv
    - .xlsx/.xls：pd.read_excel（原逻辑）
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return df.to_dict("records")


def _normalize_api_json(records: list[dict], col: str = "api_json") -> None:
    """把 api_json 字段统一为字符串（原地修改）。

    JSONL 中 api_json 可能直接写成对象/数组（更自然），而下游 build_messages /
    importer 一律用 json.loads 解析字符串。这里把 dict/list 序列化为 JSON 字符串，
    使下游零改动。已是字符串或 None 的记录保持不变。
    """
    for r in records:
        v = r.get(col)
        if isinstance(v, (dict, list)):
            r[col] = json.dumps(v, ensure_ascii=False)


def _is_blank(v) -> bool:
    """None / NaN / 空串视为空值。"""
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return False


def copy_dataset(src_path: str, dest_path: Path, columns: list[str]) -> Path:
    """复制数据集到指定路径，仅保留指定的列。

    保留源文件格式：.jsonl 源逐行读写（字段无长度损失，规避 Excel 32767 截断），
    其他源（.xlsx/.csv）维持原 pandas 读写逻辑。

    Args:
        src_path: 源数据集文件路径
        dest_path: 目标路径（目录或完整路径）
        columns: 要保留的列名列表

    Returns:
        实际写入的文件路径
    """
    src = Path(src_path)
    dest = Path(dest_path)

    # 如果 dest_path 是目录，使用源文件名（保留扩展名）
    if dest.is_dir() or not dest.suffix:
        dest = dest / src.name

    dest.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".jsonl":
        # JSONL：逐行读写，仅保留指定列，零长度损失
        cols = set(columns)
        with open(src, encoding="utf-8") as fin, \
                open(dest, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                filtered = {k: v for k, v in rec.items() if k in cols}
                fout.write(json.dumps(filtered, ensure_ascii=False) + "\n")
    elif src.suffix.lower() == ".csv":
        # CSV：无长度上限，列筛选后写回 CSV
        df = pd.read_csv(src_path)
        cols_to_keep = [c for c in columns if c in df.columns]
        df[cols_to_keep].to_csv(dest, index=False)
    else:
        # Excel：原 pandas 逻辑
        df = pd.read_excel(src_path)
        cols_to_keep = [c for c in columns if c in df.columns]
        df[cols_to_keep].to_excel(dest, index=False)

    return dest


def load_dataset(path: str) -> list[dict]:
    """从 Excel / JSONL 加载数据集，返回 list[dict]。

    必选列：query, api_json
    可选列：language, location（用于 locale 上下文注入）, domain（用于垂域评分）
    """
    records = _read_records(Path(path))
    if not records:
        return []

    # 列校验（以首行键为准）
    for col in REQUIRED_COLUMNS:
        if col not in records[0]:
            raise ValueError(f"Dataset file '{path}' must have a '{col}' column")

    # 仅保留必需 + 存在的可选列
    keep = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in records[0]]
    records = [{k: r.get(k) for k in keep} for r in records]

    # api_json 归一化为字符串（JSONL 可能写成对象/数组）
    _normalize_api_json(records)

    # 填充空值（NaN -> None）
    for r in records:
        for key in ("language", "location"):
            if _is_blank(r.get(key)):
                r[key] = None
    return records


def build_messages(api_json_str: str, system_prompt: str) -> list[dict]:
    """解析 api_json 并构建最终的 messages 列表。

    设计原则：
    - api_json 定义消息结构（roles、顺序、多轮历史）
    - system_prompt 强制覆盖 system 消息的 content（来自 prompt 文件）
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
