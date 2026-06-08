"""Import summarybox logs into a dataset ready for CLI eval.

Pipeline:
  raw_data/summarybox_log.xlsx
    → filter classification == "QA"
    → decode log_json (multi-layer JSON + HTML unescape + truncation repair)
    → extract query, answer→response, prompt→api_json
    → output data/<name>.xlsx
"""

import html
import json
from typing import Any

import pandas as pd


# ── decoder ──────────────────────────────────────────────────────────────

def _extract_msg_payload(s: str) -> str:
    pos = s.find("msg:=")
    if pos >= 0:
        return s[pos + len("msg:="):].strip()
    return s.strip()


def _repair_truncated_json(s: str) -> str:
    stack = []
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if stack:
                    stack.pop()

    if esc:
        s += "\\"
    if in_str:
        s += '"'

    pair = {"[": "]", "{": "}"}
    s += "".join(pair[x] for x in reversed(stack))
    return s


def _try_parse_string(s: str) -> Any:
    s = html.unescape(s)
    s = _extract_msg_payload(s)

    try:
        return json.loads(s)
    except Exception:
        pass

    repaired = _repair_truncated_json(s)
    try:
        return json.loads(repaired)
    except Exception:
        pass

    return s


def deep_parse(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {deep_parse(k): deep_parse(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_parse(x) for x in obj]
    if isinstance(obj, str):
        parsed = _try_parse_string(obj)
        if parsed is obj:
            return obj
        if isinstance(parsed, str):
            if parsed == obj:
                return parsed
            return deep_parse(parsed)
        return deep_parse(parsed)
    return obj


# ── import ───────────────────────────────────────────────────────────────

def import_summarybox(input_path: str, output_name: str) -> None:
    df = pd.read_excel(input_path)

    if "classification" not in df.columns:
        raise ValueError(f"Missing 'classification' column. Found: {list(df.columns)}")

    qa_rows = df[df["classification"] == "QA"]
    if qa_rows.empty:
        print("No QA rows found.")
        return

    print(f"Found {len(qa_rows)} QA row(s).")

    records = []
    decode_errors = 0
    for _, row in qa_rows.iterrows():
        raw_str = row["log_json"]
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            decode_errors += 1
            continue

        decoded = deep_parse(raw)

        gen_log = None
        for item in decoded.get("content", []):
            if "generate agg log" in item:
                gen_log = item["generate agg log"]
                break

        if gen_log is None:
            decode_errors += 1
            continue

        records.append({
            "query": gen_log.get("query", ""),
            "response": gen_log.get("answer", ""),
            "api_json": json.dumps(gen_log.get("prompt", []), ensure_ascii=False),
        })

    if not records:
        print("No valid records extracted.")
        return

    output_path = f"data/{output_name}.xlsx"
    result = pd.DataFrame(records, columns=["query", "response", "api_json"])
    result.to_excel(output_path, index=False)

    print(f"Done: {len(result)} row(s) -> {output_path}")
    if decode_errors:
        print(f"  [warn] {decode_errors} row(s) skipped due to decode errors.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import summarybox logs into eval-ready dataset.")
    parser.add_argument(
        "name", nargs="?", default="summarybox_import",
        help="Output name (written to data/<name>.xlsx, default: summarybox_import)",
    )
    parser.add_argument(
        "--input", default="raw_data/summarybox_log.xlsx",
        help="Input path (default: raw_data/summarybox_log.xlsx)",
    )
    args = parser.parse_args()
    import_summarybox(args.input, args.name)
