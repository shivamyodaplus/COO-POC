"""
ExtractionNode — uses the local VLLM model to extract structured fields from a document.

Responsibilities
----------------
1. Select an extraction prompt template based on ``doc_type``.
2. Call the LLM with the OCR text and receive a structured JSON response.
3. Write ``extracted_fields`` (dict) back into ``GraphState``.

Structured output is requested via LangChain's ``.with_structured_output()``
which uses JSON schema / function-calling under the hood.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.node_schemas import (
    ExtractionNodeInput,
    ExtractionNodeOutput,
    ExtractedField,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

# Per-doc-type field lists embedded in the prompt
_DOC_TYPE_FIELDS: dict[str, list[str]] = {
    "passport": ["full_name", "date_of_birth", "nationality", "passport_number", "expiry_date", "gender"],
    "national_id": ["full_name", "date_of_birth", "id_number", "address", "expiry_date"],
    "drivers_license": ["full_name", "date_of_birth", "license_number", "expiry_date", "categories"],
    "invoice": ["invoice_number", "invoice_date", "vendor_name", "total_amount", "currency", "due_date"],
    "bank_statement": ["account_holder", "account_number", "bank_name", "statement_period", "closing_balance"],
    "utility_bill": ["account_holder", "account_number", "provider_name", "billing_period", "amount_due"],
}

_SYSTEM_PROMPT = """\
You are a precise document data extraction assistant. Extract the specified \
fields from the provided OCR text. If a field is not present return null for it.

Respond ONLY with a JSON array where each element has:
  {{ "key": "<field_name>", "value": "<extracted_value>", "confidence": <0.0-1.0> }}
"""


def _build_prompt(doc_type: str, ocr_text: str) -> str:
    fields = _DOC_TYPE_FIELDS.get(doc_type, ["content_summary"])
    fields_list = "\n".join(f"- {f}" for f in fields)
    return (
        f"Document type: {doc_type}\n"
        f"Fields to extract:\n{fields_list}\n\n"
        f"OCR text:\n\n{ocr_text[:4000]}\n\n"
        "Extract the fields as a JSON array."
    )


def extraction_node(state: GraphState) -> GraphState:
    """LangGraph node: extract structured fields from the document via LLM.

    Reads
    -----
    state["doc_type"]   — determines which fields to extract.
    state["messages"]   — last HumanMessage is treated as OCR text.
    state["document_id"]

    Writes
    ------
    state["extracted_fields"], state["current_step"]
    Appends LLM response to state["messages"].
    """
    logger.info("extraction_node: starting")

    errors: list[str] = list(state.get("errors") or [])

    # Extract OCR text from messages
    messages = state.get("messages") or []
    ocr_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            ocr_text = str(msg.content)
            break

    if not ocr_text:
        errors.append("extraction_node: no OCR text found in messages")
        return {**state, "errors": errors, "current_step": "extraction_node"}  # type: ignore[return-value]

    doc_type = state.get("doc_type") or "other"

    # --- Validate input via Pydantic ----------------------------------------
    try:
        node_input = ExtractionNodeInput(
            document_id=state.get("document_id") or "",
            doc_type=doc_type,
            ocr_text=ocr_text,
        )
    except Exception as exc:
        errors.append(f"extraction_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "extraction_node"}  # type: ignore[return-value]

    # --- LLM call -----------------------------------------------------------
    llm = get_structured_llm(ExtractionNodeOutput, temperature=0.0)
    prompt_messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_prompt(node_input.doc_type, node_input.ocr_text)),
    ]

    try:
        node_output: ExtractionNodeOutput = llm.invoke(prompt_messages)  # type: ignore[assignment]
    except Exception as exc:
        errors.append(f"extraction_node: LLM call failed: {exc}")
        return {**state, "errors": errors, "current_step": "extraction_node"}  # type: ignore[return-value]

    # Flatten to dict for the shared state
    fields_dict = {
        f.key: f.value
        for f in node_output.extracted_fields
        if f.value is not None
    }

    logger.info("extraction_node: extracted %d fields", len(fields_dict))

    return {  # type: ignore[return-value]
        **state,
        "extracted_fields": fields_dict,
        "errors": errors,
        "current_step": "extraction_node",
    }
