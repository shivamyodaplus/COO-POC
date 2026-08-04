"""
ClassificationNode — uses the local VLLM model to identify country + document type.

Responsibilities
----------------
1. Build a classification prompt from the OCR text in the state.
2. Call the VLLM-backed LLM via LangChain's structured output API.
3. Write ``country`` and ``doc_type`` back into ``GraphState``.

The node expects OCR text to already be present in ``state["messages"]``
(added by an OCR pre-processing step or the ingestion node).
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_llm
from app.langgraph.schemas.node_schemas import (
    ClassificationNodeInput,
    ClassificationNodeOutput,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a document classification expert. Given OCR text from a document, \
identify the issuing country (ISO 3166-1 alpha-2 code) and the document type.

Respond ONLY with a JSON object matching this schema:
{
  "country": "<ISO code>",
  "doc_type": "<type>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}

Supported document types: passport, national_id, drivers_license, invoice, \
bank_statement, utility_bill, other.
"""


def _build_prompt(ocr_text: str, country_hint: str | None) -> str:
    hint_line = f"\nCountry hint: {country_hint}" if country_hint else ""
    return f"OCR text:\n\n{ocr_text[:4000]}{hint_line}\n\nClassify the document."


def classification_node(state: GraphState) -> GraphState:
    """LangGraph node: classify document country and type via LLM.

    Reads
    -----
    state["messages"]  — last HumanMessage content is treated as OCR text.
    state["country"]   — optional hint forwarded to the LLM.

    Writes
    ------
    state["country"], state["doc_type"], state["current_step"]
    Appends to state["messages"] with the assistant reply.
    """
    logger.info("classification_node: starting")

    errors: list[str] = list(state.get("errors") or [])

    # Extract OCR text: prefer explicit state field, else last human message
    messages = state.get("messages") or []
    ocr_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            ocr_text = str(msg.content)
            break

    if not ocr_text:
        errors.append("classification_node: no OCR text found in messages")
        return {**state, "errors": errors, "current_step": "classification_node"}  # type: ignore[return-value]

    # --- Validate input via Pydantic ----------------------------------------
    try:
        node_input = ClassificationNodeInput(
            document_id=state.get("document_id") or "",
            ocr_text=ocr_text,
            country_hint=state.get("country"),
        )
    except Exception as exc:
        errors.append(f"classification_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "classification_node"}  # type: ignore[return-value]

    # --- LLM call -----------------------------------------------------------
    llm = get_llm(temperature=0.0)
    prompt_messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_prompt(node_input.ocr_text, node_input.country_hint)),
    ]

    try:
        response = llm.invoke(prompt_messages)
        raw_response = str(response.content)
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        errors.append(f"classification_node: LLM returned non-JSON: {raw_response[:200]}")
        return {**state, "errors": errors, "current_step": "classification_node"}  # type: ignore[return-value]
    except Exception as exc:
        errors.append(f"classification_node: LLM call failed: {exc}")
        return {**state, "errors": errors, "current_step": "classification_node"}  # type: ignore[return-value]

    # --- Validate output via Pydantic ----------------------------------------
    try:
        node_output = ClassificationNodeOutput(**parsed)
    except Exception as exc:
        errors.append(f"classification_node output schema error: {exc}")
        return {**state, "errors": errors, "current_step": "classification_node"}  # type: ignore[return-value]

    logger.info(
        "classification_node: country=%s doc_type=%s confidence=%.2f",
        node_output.country,
        node_output.doc_type,
        node_output.confidence,
    )

    return {  # type: ignore[return-value]
        **state,
        "country": node_output.country,
        "doc_type": node_output.doc_type,
        "messages": [response],  # add_messages reducer will append
        "errors": errors,
        "current_step": "classification_node",
    }
