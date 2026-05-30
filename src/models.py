"""模型客户端模块。

封装 OpenAI 兼容接口，统一调用方式。
非标准 API 参数（如 chat_template_kwargs）通过 extra_body 传递给模型。
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


async def call_model(client: AsyncOpenAI, model_config: ModelConfig, request: dict) -> tuple[dict, dict]:
    """调用模型 API。

    Args:
        client: AsyncOpenAI 客户端实例
        model_config: 模型配置，call_params 作为默认值（优先级高于 request）
        request: 渲染后的 API 请求 dict，model_config 中未定义的字段可由此补充

    Returns:
        (response_dict, actual_kwargs): API 响应 + 实际发送的完整请求参数
    """
    kwargs = {**request, **model_config.call_params}

    response = await client.chat.completions.create(**kwargs)
    return response.model_dump(), kwargs
