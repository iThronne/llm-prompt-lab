"""模型客户端模块。

封装 OpenAI 兼容接口，统一调用方式。
非标准 API 参数（如 chat_template_kwargs）通过 extra_body 传递给模型。
支持通过 use_proxy 启用代理 + 禁用 SSL 验证（代理地址从环境变量 PROXY_URL 读取）。
"""

import os
import time

import httpx
from openai import AsyncOpenAI

from src.config import ModelConfig


def create_client(model_config: ModelConfig) -> AsyncOpenAI:
    """根据模型配置创建 AsyncOpenAI 客户端。

    当 use_proxy=True 时，从环境变量 PROXY_URL 读取代理地址，
    创建 httpx.AsyncClient 并禁用 SSL 验证，传递给 AsyncOpenAI。
    """
    api_key = ""
    if model_config.api_key_env:
        api_key = os.getenv(model_config.api_key_env, "")

    kwargs: dict = {"api_key": api_key, "base_url": model_config.base_url}

    if model_config.use_proxy:
        proxy_url = os.getenv("PROXY_URL", "")
        if not proxy_url:
            raise ValueError("use_proxy=True but PROXY_URL is not set in environment")
        ssl_context = httpx.create_ssl_context(verify=False)  # 禁用 SSL 验证
        http_client = httpx.AsyncClient(
            proxy=proxy_url,
            verify=ssl_context,  # 强行把“禁用 SLL 验证”的上下文塞给客户端
        )
        kwargs["http_client"] = http_client

    return AsyncOpenAI(**kwargs)


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


async def call_model_stream(
    client: AsyncOpenAI, model_config: ModelConfig, request: dict,
) -> tuple[dict, dict, float | None]:
    """流式调用模型 API，逐 chunk 收集内容并组装为与非流式相同的响应格式。

    额外返回 TTFT（首 token 延迟，毫秒），用于性能分析。

    Returns:
        (response_dict, actual_kwargs, ttft_ms)
    """
    kwargs = {**request, **model_config.call_params, "stream": True}

    start_time = time.monotonic()
    stream = await client.chat.completions.create(**kwargs)

    chunks = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    first_token_time: float | None = None
    finish_reason: str | None = None

    async for chunk in stream:
        chunks.append(chunk)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # 提取推理内容（思考模型的扩展字段，各家命名不同）
        if delta.model_extra:
            reasoning = delta.model_extra.get("reasoning") or delta.model_extra.get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)

        if delta.content:
            if first_token_time is None:
                first_token_time = time.monotonic()
            content_parts.append(delta.content)
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    # 组装为与 chat.completions.create() 非流式响应相同的 dict 结构
    last = chunks[-1] if chunks else None
    message = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    # 如果有推理内容，添加到 message 中（兼容各家扩展字段）
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)

    response_dict = {
        "id": last.id if last else "",
        "object": "chat.completion",
        "created": last.created if last else 0,
        "model": last.model if last else "",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": last.usage.model_dump() if last and last.usage else None,
    }

    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else None
    return response_dict, kwargs, ttft_ms
