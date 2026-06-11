"""Excel 第一列翻译脚本。

读取 Excel 的第一列（中文），调用大模型翻译为英文，写入第二列。
保持原有格式和语义不变。支持断点续跑（已有翻译的行自动跳过）。
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


def create_client() -> OpenAI:
    """从环境变量读取配置，创建 OpenAI 客户端。"""
    api_key = os.getenv("DEFAULT_OPENAI_API_KEY", "")
    base_url = os.getenv("DEFAULT_OPENAI_API_URL", "")
    if not api_key:
        raise ValueError("请在 .env 中设置 DEFAULT_OPENAI_API_KEY")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


SYSTEM_PROMPT = """You are a professional translator. Your task is to translate Chinese text to English.

Rules:
1. Translate the given text into natural, idiomatic English.
2. Preserve the original meaning and tone exactly — do not add, remove, or alter any information.
3. Keep the same style: formal stays formal, casual stays casual, technical terms stay technical.
4. Output ONLY the English translation, nothing else — no explanations, no notes, no quotation marks.
5. If the input is already in English, return it unchanged."""


def translate_text(client: OpenAI, text: str, model: str) -> str:
    """调用大模型翻译单条文本。

    Args:
        client: OpenAI 客户端
        text: 待翻译的中文文本
        model: 模型名称

    Returns:
        英文翻译结果
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="将 Excel 第一列翻译为英文，写入第二列",
    )
    parser.add_argument("excel", help="Excel 文件路径")
    parser.add_argument(
        "--col", type=int, default=0,
        help="要翻译的列索引（0-based，默认 0 即第一列）",
    )
    parser.add_argument(
        "--target-col", type=int, default=1,
        help="写入翻译结果的列索引（0-based，默认 1 即第二列）",
    )
    parser.add_argument(
        "--model", default="deepseek-v4-flash",
        help="模型名称（默认 deepseek-v4-flash）",
    )
    parser.add_argument(
        "--output", "-o",
        help="输出文件路径（默认覆盖原文件，加此参数则输出到新文件）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新翻译所有行（忽略已有的翻译结果）",
    )
    parser.add_argument(
        "--skip-empty", action="store_true", default=True,
        help="跳过空单元格（默认开启）",
    )
    parser.add_argument(
        "--save-interval", type=int, default=20,
        help="每翻译 N 行保存一次（默认 20），防止中断丢失进度",
    )

    args = parser.parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"[error] 文件不存在: {excel_path}")
        sys.exit(1)

    # 读取 Excel
    print(f"[info] 读取文件: {excel_path}")
    df = pd.read_excel(excel_path)

    col_idx = args.col
    target_col_idx = args.target_col

    if col_idx >= len(df.columns):
        print(f"[error] 列索引 {col_idx} 超出范围（共 {len(df.columns)} 列）")
        sys.exit(1)

    col_name = df.columns[col_idx]

    # 确保目标列存在
    if target_col_idx >= len(df.columns):
        # 扩展列
        for i in range(len(df.columns), target_col_idx + 1):
            df[f"Column_{i + 1}"] = None

    target_col_name = df.columns[target_col_idx]

    # 筛选需要翻译的行
    texts_to_translate = []
    indices_to_translate = []

    for i, row in df.iterrows():
        source_text = row.iloc[col_idx]

        # 跳过空值
        if pd.isna(source_text) or str(source_text).strip() == "":
            if args.skip_empty:
                continue

        source_text = str(source_text).strip()

        # 断点续跑：已有翻译的行跳过
        existing = row.iloc[target_col_idx]
        if not args.force and not pd.isna(existing) and str(existing).strip() != "":
            continue

        texts_to_translate.append(source_text)
        indices_to_translate.append(i)

    if not texts_to_translate:
        print("[info] 没有需要翻译的行（所有行已有翻译结果）")
        return

    print(f"[info] 共 {len(texts_to_translate)} 行需要翻译")

    # 创建客户端并翻译
    client = create_client()
    model = args.model
    save_interval = args.save_interval
    output_path = Path(args.output) if args.output else excel_path

    total = len(texts_to_translate)
    translated_count = 0

    pbar = tqdm(total=total, desc="翻译中")
    for batch_start in range(0, total, save_interval):
        batch_end = min(batch_start + save_interval, total)
        batch_texts = texts_to_translate[batch_start:batch_end]
        batch_indices = indices_to_translate[batch_start:batch_end]

        for text, idx in zip(batch_texts, batch_indices):
            try:
                result = translate_text(client, text, model)
                df.iloc[idx, target_col_idx] = result
            except Exception as e:
                print(f"\n[warn] 翻译失败 (行 {idx}): {text[:50]}... → {e}")
                df.iloc[idx, target_col_idx] = f"[翻译失败] {text}"
            translated_count += 1
            pbar.update(1)

        # 每批保存一次，防止中断丢失进度
        df.to_excel(output_path, index=False)

    pbar.close()
    print(f"[done] 翻译完成，共处理 {translated_count} 行 → {output_path}")


if __name__ == "__main__":
    main()
