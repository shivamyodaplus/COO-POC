"""
LLM client factory — supports local VLLM and AWS Bedrock, selected via
the ``LLM_PROVIDER`` environment variable.

Provider: ``local`` (default)
    Wraps the local VLLM server via LangChain's ChatOpenAI.  VLLM exposes an
    OpenAI-compatible REST API; ``base_url`` is pointed at the local endpoint
    and a dummy API key is passed (VLLM ignores it).

Provider: ``bedrock``
    Uses ChatBedrockConverse from langchain-aws, targeting AWS Bedrock.
    AWS credentials are resolved by boto3's standard credential chain
    (env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN,
    ~/.aws/credentials, IAM instance role, etc.).

Two factories are provided:
  get_llm()        – text-only model (classification, extraction, report generation)
  get_vision_llm() – vision/multimodal model (image + text input)

All settings are pulled from app.core.config.settings so they are unified
under the same .env file.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """Return a chat model for text tasks, dispatched by LLM_PROVIDER."""
    _temp = temperature if temperature is not None else settings.VLLM_TEMPERATURE
    _max_tokens = max_tokens if max_tokens is not None else settings.VLLM_MAX_TOKENS

    if settings.LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model_id=settings.BEDROCK_TEXT_MODEL,
            region_name=settings.BEDROCK_REGION,
            temperature=_temp,
            max_tokens=_max_tokens if _max_tokens is not None else settings.BEDROCK_MAX_TOKENS,
            streaming=streaming,
            # Disable Qwen3 extended thinking on Bedrock so output is plain text.
            additional_model_request_fields={"thinking": {"type": "disabled"}},
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


def get_vision_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """Return a chat model for vision/multimodal tasks, dispatched by LLM_PROVIDER.

    Both providers accept messages with image_url content parts encoded as
    base64 data URIs — use ``encode_image_for_llm()`` to produce them.
    """
    _temp = temperature if temperature is not None else settings.VLLM_TEMPERATURE
    _max_tokens = max_tokens if max_tokens is not None else settings.VLLM_MAX_TOKENS

    if settings.LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model_id=settings.BEDROCK_VISION_MODEL,
            region_name=settings.BEDROCK_REGION,
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


def encode_image_for_llm(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode raw image bytes as a base64 data URI for use in vision LLM messages."""
    import base64
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"
