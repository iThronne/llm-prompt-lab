"""脱敏模块。

在把候选模型的 messages 传给 judge 大模型之前，按 eval.yaml 中配置的
正则规则替换敏感信息（手机号、身份证、邮箱、API key 等）。

设计原则：
- 规则在 evaluator 启动时编译一次，避免每条数据重复编译。
- sanitize_messages 深拷贝原 messages，不污染调用方的数据。
- 递归处理 content（字符串 / multimodal list）、tool_calls.function.arguments、
  tool role 的 content 等所有可能携带原始用户文本的位置。
"""

import copy
import re
from typing import Any

from src.config import SanitizeRule

CompiledRules = list[tuple[re.Pattern, str]]


def compile_rules(rules: list[SanitizeRule]) -> CompiledRules:
    """编译脱敏规则。规则中正则非法时直接抛错，提示使用者修复 eval.yaml。"""
    compiled: CompiledRules = []
    for rule in rules:
        try:
            compiled.append((re.compile(rule.pattern), rule.replacement))
        except re.error as e:
            raise ValueError(f"sanitize 规则正则非法: {rule.pattern} ({e})")
    return compiled


def sanitize_text(text: str, compiled: CompiledRules) -> str:
    """对单段文本应用所有脱敏规则。"""
    if not text or not compiled:
        return text
    for pat, repl in compiled:
        text = pat.sub(repl, text)
    return text


def _sanitize_content(content: Any, compiled: CompiledRules) -> Any:
    """递归处理 message.content，兼容字符串和 multimodal list 形式。"""
    if isinstance(content, str):
        return sanitize_text(content, compiled)
    if isinstance(content, list):
        result = []
        for part in content:
            if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                new_part = dict(part)
                new_part["text"] = sanitize_text(part["text"], compiled)
                result.append(new_part)
            else:
                result.append(part)
        return result
    return content


def sanitize_messages(messages: list[dict], compiled: CompiledRules) -> list[dict]:
    """对完整 messages 列表执行脱敏，返回深拷贝后的新列表。

    覆盖位置：
    - 任意 role 的 content（字符串或 multimodal）
    - assistant.tool_calls[*].function.arguments（JSON 字符串）
    - tool.content（搜索结果文本）
    """
    if not compiled:
        return copy.deepcopy(messages)

    sanitized: list[dict] = []
    for msg in messages:
        new_msg = copy.deepcopy(msg)

        if "content" in new_msg:
            new_msg["content"] = _sanitize_content(new_msg["content"], compiled)

        tool_calls = new_msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                func = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(func, dict) and isinstance(func.get("arguments"), str):
                    func["arguments"] = sanitize_text(func["arguments"], compiled)

        sanitized.append(new_msg)
    return sanitized
