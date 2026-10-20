"""提取生产日志中内嵌的搜索专项视图，不改写原始 messages。"""

import re


def embedded_search_contexts(messages: list) -> list[dict]:
    """只读取 user 中完整的 web_search_context，保留证据所在字符串路径。"""
    result = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        texts = [(f"/messages/{index}/content", content)] if isinstance(content, str) else []
        if isinstance(content, list):
            texts = [(f"/messages/{index}/content/{i}/text", part["text"])
                     for i, part in enumerate(content)
                     if isinstance(part, dict) and isinstance(part.get("text"), str)]
        for path, text in texts:
            for block in re.finditer(r"<web_search_context>(.*?)</web_search_context>", text, re.S):
                entry = {"path": path}
                for tag in ("search_queries", "search_results"):
                    entry[tag] = "\n\n".join(
                        match.group(1).strip()
                        for match in re.finditer(fr"<{tag}>(.*?)</{tag}>", block.group(1), re.S)
                        if match.group(1).strip()
                    )
                if entry["search_queries"] or entry["search_results"]:
                    result.append(entry)
    return result
