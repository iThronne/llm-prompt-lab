"""仅生成匹配键，不改变保存的 Query、评论或证据原文。

与 templates/text_matching.js 保持一致，并由跨语言测试约束。
"""

import re
import unicodedata

_SPACES = re.compile(r"[\t\n\v\f\r \u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")
_MARKS = str.maketrans({"？": "?", "﹖": "?", "！": "!", "﹗": "!",
                       "\u200b": None, "\ufeff": None, "\u2060": None})


def exact_query(value):
    return str(value if value is not None else "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_query(value):
    text = unicodedata.normalize("NFC", str(value if value is not None else "")).translate(_MARKS)
    text = _SPACES.sub(" ", text).strip()
    return re.sub(r" +([?!])", r"\1", text)


def question_key(value):
    """仅弱化句首倒问号、句末问号，保留内部问号及其他语义字符。"""
    return re.sub(r"\?+$", "", re.sub(r"^¿+", "", normalize_query(value))).strip()
