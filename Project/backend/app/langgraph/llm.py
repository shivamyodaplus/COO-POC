"""
LLM client factory — wraps the local VLLM server via LangChain's ChatOpenAI.

VLLM exposes an OpenAI-compatible REST API, so we point ``base_url`` at the
local inference endpoint and pass a dummy API key (VLLM ignores it).

Two factories are provided:
  get_llm()        – text-only model (classification, extraction, report generation)
  get_vision_llm() – vision/multimodal model (image + text input)

All settings are pulled from app.core.config.settings so they are unified
under the same .env file.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """Return a ChatOpenAI client targeting the local VLLM text model."""
    return ChatOpenAI(
        base_url=settings.VLLM_BASE_URL,
        model=settings.VLLM_TEXT_MODEL,
        api_key=settings.VLLM_API_KEY,  # type: ignore[arg-type]
        temperature=temperature if temperature is not None else settings.VLLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else settings.VLLM_MAX_TOKENS,
        streaming=streaming,
    )


def get_vision_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """Return a ChatOpenAI client targeting the local VLLM vision/multimodal model.

    The vision model accepts messages with image_url content parts, which are
    encoded as base64 data URIs when calling ``encode_image_for_llm()``.
    """
    return ChatOpenAI(
        base_url=settings.VLLM_BASE_URL,
        model=settings.VLLM_VISION_MODEL,
        api_key=settings.VLLM_API_KEY,  # type: ignore[arg-type]
        temperature=temperature if temperature is not None else settings.VLLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else settings.VLLM_MAX_TOKENS,
        streaming=streaming,
    )


def encode_image_for_llm(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode raw image bytes as a base64 data URI for use in vision LLM messages."""
    import base64
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"
