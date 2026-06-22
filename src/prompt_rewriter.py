"""Prompt 反推与重渲染模块。

用途：生产 api_json[system].content 是用"原模板 A（jinja2）+ 真实 context"渲染出来的。
本模块把"原模板 A + 已渲染串"反推出 context dict，然后用该 context 渲染"实验模板 B"，
让实验保留真实样本上下文，仅替换 prompt 主体。

约束：模板只能含 {{ var }} 纯变量替换，不支持 filter / {% if %} / {% for %} / 宏。
"""

import re

from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import UndefinedError
from jinja2.nodes import Name, Output, TemplateData


def extract_context(source_template: str, rendered: str) -> dict[str, str]:
    """用 jinja2 AST 把 source_template 拆成 [字面量, 变量, ...]，转正则回填 rendered。

    Args:
        source_template: 原模板字符串（仅含 {{ var }} 占位符）
        rendered: 已渲染的成品串（即生产 api_json[system].content）

    Returns:
        {变量名: 提取到的字符串值}

    Raises:
        NotImplementedError: 模板含不支持的节点（filter / if / for 等）
        ValueError: 正则匹配失败（模板与渲染串不匹配）
    """
    env = Environment()
    ast = env.parse(source_template)

    # 遍历 AST：顶层应为若干 Output 节点，每个 Output 包含 TemplateData / Name 子节点
    parts: list[tuple[str, str]] = []  # (kind, payload)，kind ∈ {"lit", "var"}
    for node in ast.body:
        if not isinstance(node, Output):
            raise NotImplementedError(
                f"不支持的模板节点: {type(node).__name__}（仅支持 {{{{ var }}}} 纯变量替换）"
            )
        for child in node.nodes:
            if isinstance(child, TemplateData):
                parts.append(("lit", child.data))
            elif isinstance(child, Name):
                parts.append(("var", child.name))
            else:
                raise NotImplementedError(
                    f"不支持的模板子节点: {type(child).__name__}（仅支持字面量与变量名）"
                )

    # 拼正则
    pattern_chunks: list[str] = []
    seen_vars: set[str] = set()
    for kind, payload in parts:
        if kind == "lit":
            pattern_chunks.append(re.escape(payload))
        else:
            # 同名变量在模板里出现多次时，第二次起用 backreference 保证一致
            if payload in seen_vars:
                pattern_chunks.append(f"(?P={payload})")
            else:
                pattern_chunks.append(f"(?P<{payload}>.*?)")
                seen_vars.add(payload)

    pattern = "".join(pattern_chunks)
    match = re.fullmatch(pattern, rendered, flags=re.DOTALL)
    if match is None:
        raise ValueError("source_template 与 rendered 串不匹配，无法回填 context")
    return match.groupdict()


def render_with_context(target_template: str, context: dict) -> str:
    """用 jinja2 渲染目标模板。

    使用 StrictUndefined，模板中存在但 context 缺失的变量会抛 UndefinedError，
    避免变量名不一致时静默产出空串。
    """
    env = Environment(undefined=StrictUndefined)
    template = env.from_string(target_template)
    try:
        return template.render(**context)
    except UndefinedError as e:
        # 转成 ValueError，让调用方用统一的 except 处理
        raise ValueError(f"渲染失败：模板含 context 中不存在的变量 ({e})") from e
