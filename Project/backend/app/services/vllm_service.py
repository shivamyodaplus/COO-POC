from __future__ import annotations

import base64
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _text_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=settings.QWEN_TEXT_BASE_URL, api_key="none")


def _vl_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=settings.QWEN_VL_BASE_URL, api_key="none")


def _health_url(base_url: str) -> str:
    """Return the /health endpoint from an OpenAI-compatible base URL like http://host:port/v1."""
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


async def chat_text(
    messages: list[dict[str, Any]],
    *,
    model: str = "qwen3-4b-awq",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    """Send a text-only chat request to the qwen-text endpoint.

    Qwen3 thinking mode is disabled by default to avoid burning the token
    budget on a <think> block and returning an empty visible answer.
    """
    response = await _text_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        **kwargs,
    )
    return response.choices[0].message.content or ""


async def chat_vision(
    image_bytes: bytes,
    prompt: str,
    *,
    model: str = "qwen3-vl-4b",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    """Send an image + text prompt to the qwen-vl endpoint.

    Args:
        image_bytes: Raw image bytes (PNG or JPEG).
        prompt: Text instruction to accompany the image.
    Returns:
        Model response as a string.
    """
    b64 = base64.b64encode(image_bytes).decode()
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }
    ]
    response = await _vl_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""


async def check_health() -> dict[str, bool]:
    """Check /health on both vLLM endpoints.

    Returns a dict like {"qwen-text": True, "qwen-vl": False}.
    Never raises — unreachable endpoints are reported as False.
    """
    endpoints = {
        "qwen-text": _health_url(settings.QWEN_TEXT_BASE_URL),
        "qwen-vl": _health_url(settings.QWEN_VL_BASE_URL),
    }
    results: dict[str, bool] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in endpoints.items():
            try:
                resp = await client.get(url)
                results[name] = resp.status_code == 200
            except httpx.RequestError:
                results[name] = False
    return results
