"""
LLM client factory — supports local VLLM and AWS Bedrock, selected via
the ``LLM_PROVIDER`` environment variable.

Provider: ``local`` (default)
    Wraps the local VLLM server via LangChain's ChatOpenAI.  VLLM exposes an
    OpenAI-compatible REST API; ``base_url`` is pointed at the local endpoint
    and a dummy API key is passed (VLLM ignores it).

Provider: ``bedrock``
    Uses ChatOpenAI pointed at Bedrock's OpenAI-compatible Chat Completions
    endpoint (``https://bedrock-mantle.<region>.api.aws/v1``) with a short-term
    Bedrock API key supplied via ``BEDROCK_API_KEY``.  No boto3 or IAM
    credentials are required.

Factories are cached via ``@lru_cache`` so identical parameter combinations
return the same client instance across the entire application lifecycle.

  get_llm()            – cached text-only model
  get_vision_llm()     – cached vision/multimodal model
  get_structured_llm() – cached structured-output runnable (schema-bound)

All settings are pulled from app.core.config.settings so they are unified
under the same .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings


# ──────────────────────────────────────────────────────────────────────────────
# Public text LLM factory (keyword-only interface → delegates to cached impl)
# ──────────────────────────────────────────────────────────────────────────────


def get_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """Return a cached chat model for text tasks, dispatched by LLM_PROVIDER."""
    return _get_llm_cached(temperature, max_tokens, streaming)


@lru_cache(maxsize=8)
def _get_llm_cached(
    temperature: float | None,
    max_tokens: int | None,
    streaming: bool,
) -> BaseChatModel:
    _temp = temperature if temperature is not None else settings.VLLM_TEMPERATURE
    _max_tokens = max_tokens if max_tokens is not None else settings.VLLM_MAX_TOKENS

    if settings.LLM_PROVIDER == "bedrock":
        # api_key and base_url are read automatically from OPENAI_API_KEY
        # and OPENAI_BASE_URL environment variables (AWS Bedrock OpenAI pattern).
        return ChatOpenAI(
            model=settings.BEDROCK_TEXT_MODEL,
            temperature=_temp,
            max_tokens=_max_tokens if _max_tokens is not None else settings.BEDROCK_MAX_TOKENS,
            streaming=streaming,
        )

    return ChatOpenAI(
        base_url=settings.QWEN_TEXT_BASE_URL,
        model=settings.VLLM_TEXT_MODEL,
        api_key=settings.VLLM_API_KEY,  # type: ignore[arg-type]
        temperature=_temp,
        max_tokens=_max_tokens,
        streaming=streaming,
        # Disable Qwen3 chain-of-thought prefix (<think>…</think>).
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public vision LLM factory
# ──────────────────────────────────────────────────────────────────────────────


def get_vision_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """Return a cached chat model for vision/multimodal tasks, dispatched by LLM_PROVIDER.

    Both providers accept messages with image_url content parts encoded as
    base64 data URIs — use ``encode_image_for_llm()`` to produce them.
    """
    return _get_vision_llm_cached(temperature, max_tokens, streaming)


@lru_cache(maxsize=8)
def _get_vision_llm_cached(
    temperature: float | None,
    max_tokens: int | None,
    streaming: bool,
) -> BaseChatModel:
    _temp = temperature if temperature is not None else settings.VLLM_TEMPERATURE
    _max_tokens = max_tokens if max_tokens is not None else settings.VLLM_MAX_TOKENS

    if settings.LLM_PROVIDER == "bedrock":
        # api_key and base_url are read automatically from OPENAI_API_KEY
        # and OPENAI_BASE_URL environment variables (AWS Bedrock OpenAI pattern).
        return ChatOpenAI(
            model=settings.BEDROCK_VISION_MODEL,
            temperature=_temp,
            max_tokens=_max_tokens if _max_tokens is not None else settings.BEDROCK_MAX_TOKENS,
            streaming=streaming,
        )

    return ChatOpenAI(
        base_url=settings.QWEN_VL_BASE_URL,
        model=settings.VLLM_VISION_MODEL,
        api_key=settings.VLLM_API_KEY,  # type: ignore[arg-type]
        temperature=_temp,
        max_tokens=_max_tokens,
        streaming=streaming,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Structured output factory (cached per schema + LLM config)
# ──────────────────────────────────────────────────────────────────────────────


def get_structured_llm(
    schema: Type[BaseModel],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    vision: bool = False,
) -> Runnable:
    """Return a cached structured-output runnable bound to the given Pydantic schema.

    Usage::

        llm = get_structured_llm(MySchema, temperature=0.0)
        result: MySchema = llm.invoke([message])

    The same (schema, temperature, max_tokens, vision) combination always
    returns the identical Runnable instance.
    """
    return _get_structured_llm_cached(schema, temperature, max_tokens, vision)


@lru_cache(maxsize=16)
def _get_structured_llm_cached(
    schema: Type[BaseModel],
    temperature: float | None,
    max_tokens: int | None,
    is_vision: bool,
) -> Runnable:
    if is_vision:
        base = get_vision_llm(temperature=temperature, max_tokens=max_tokens)
    else:
        base = get_llm(temperature=temperature, max_tokens=max_tokens)
    return base.with_structured_output(schema)


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────


_MAX_VISION_LONG_EDGE = 1024  # pixels — caps visual tokens to ~960 (vs ~2500 for scale=2 PDFs)


def encode_image_for_llm(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    max_long_edge: int = _MAX_VISION_LONG_EDGE,
) -> str:
    """Encode raw image bytes as a base64 data URI for use in vision LLM messages.

    Automatically resizes the image so its longest edge does not exceed
    ``max_long_edge`` pixels before encoding.  This keeps visual token counts
    within the vision model's context window regardless of the source DPI.
    Pass ``max_long_edge=0`` to skip resizing.
    """
    import base64
    import io

    from PIL import Image

    if max_long_edge > 0:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()
            mime_type = "image/jpeg"

    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"
