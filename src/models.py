"""模型客户端模块。

封装 OpenAI 兼容接口，统一调用方式。
"""

import os
from openai import AsyncOpenAI

from src.config import ModelConfig


def create_client(model_config: ModelConfig) -> AsyncOpenAI:
    """根据模型配置创建 AsyncOpenAI 客户端。"""
    api_key = ""
    if model_config.api_key_env:
        api_key = os.getenv(model_config.api_key_env, "")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=model_config.base_url,
    )


async def call_model(client: AsyncOpenAI, model_name: str, request: dict, params: dict) -> dict:
    """调用模型 API。

    Args:
        client: AsyncOpenAI 客户端实例
        model_name: 模型名（覆盖 request 中的 model 字段）
        request: 渲染后的 API 请求 dict，包含 messages 等
        params: 额外参数（temperature, max_tokens 等）

    Returns:
        完整的 API 响应 dict
    """
    # 构建请求参数，request 中的字段优先，params 做默认覆盖
    kwargs = {**params, "model": model_name}

    # 合并 request 中的字段（messages, tools 等）
    for key, value in request.items():
        if key != "model":
            kwargs[key] = value

    response = await client.chat.completions.create(**kwargs)
    return response.model_dump()