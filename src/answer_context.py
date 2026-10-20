"""回答表示的评测边界；引用产物仅存档，暂不参与引用核验。"""

PLAIN_RESPONSE_NOTICE = (
    "当前评估的是去除引用标记后的生产 response，原始模型 Answer 不可得。"
    "response_with_cites 和 cites 仅存档，本阶段不评显式引用准确性。"
    "不得因缺少引用标记扣分，不得认定模型未引用、引用编号错误或违反引用格式要求。"
    "仍可根据正文与完整搜索材料评估事实、信息利用和可观察来源；"
    "这种文本对应关系不是恢复出的原始引用关系。"
)


def answer_notice(row: dict) -> str:
    if row.get("answer_representation") == "production_plain_response":
        return PLAIN_RESPONSE_NOTICE
    return ""
