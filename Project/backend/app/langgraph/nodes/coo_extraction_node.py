"""
COOExtractionNode — Vision LLM extracts structured field values from the COO
document using the confirmed template's field schema as a guided extraction prompt.

If no template_attributes are available (template confirmation failed), the node
falls back to general key-value extraction.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from app.langgraph.llm import encode_image_for_llm, get_vision_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    COOExtractionInput,
    COOExtractionOutput,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_GUIDED = """\
You are a precise document data extraction assistant.
Extract the following labeled fields from the COO (Certificate of Origin) document image.

Fields to extract:
{field_list}

Respond ONLY with a JSON object where keys are the field names and values are the
extracted string values.  Use null for fields not found.

Example:
{{ "exporter_name": "ABC Trading Co.", "country_of_origin": "Egypt" }}
"""

_SYSTEM_PROMPT_GENERAL = """\
You are a precise document data extraction assistant.
Extract ALL labeled key-value information from this COO (Certificate of Origin) document image.

Respond ONLY with a JSON object where keys are field names in snake_case and values are the
extracted string values.

Return {} if nothing can be extracted.
"""


def coo_extraction_node(state: GraphState) -> GraphState:
    """LangGraph node: extract COO field values via Vision LLM.

    Reads
    -----
    state["page_images"]         — COO page images (first used)
    state["template_attributes"] — field schema from confirmed template
    state["document_id"]

    Writes
    ------
    state["coo_extracted_fields"]  — {field_key: value} flat dict
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("coo_extraction_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    page_images: list[bytes] = state.get("page_images") or []
    template_attributes: dict = state.get("template_attributes") or {}

    if not page_images:
        errors.append("coo_extraction_node: no page_images in state")
        return {**state, "errors": errors, "current_step": "coo_extraction_node"}  # type: ignore[return-value]

    try:
        node_input = COOExtractionInput(
            document_id=state.get("document_id") or "",
            coo_image_b64=encode_image_for_llm(page_images[0], "image/jpeg"),
            template_attributes=template_attributes,
        )
    except Exception as exc:
        errors.append(f"coo_extraction_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "coo_extraction_node"}  # type: ignore[return-value]

    if template_attributes:
        field_list = "\n".join(f"- {k}: {v}" for k, v in template_attributes.items())
        system_text = _SYSTEM_PROMPT_GUIDED.format(field_list=field_list)
    else:
        system_text = _SYSTEM_PROMPT_GENERAL

    llm = get_vision_llm(temperature=0.0)
    message = HumanMessage(
        content=[
            {"type": "text", "text": system_text},
            {
                "type": "image_url",
                "image_url": {"url": node_input.coo_image_b64},
            },
        ]
    )

    raw_response = ""
    coo_fields: dict[str, str] = {}

    try:
        response = llm.invoke([message])
        raw_response = str(response.content)
        parsed = json.loads(raw_response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        coo_fields = {k: str(v) for k, v in parsed.items() if v is not None}
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"coo_extraction_node: bad LLM JSON — {exc}: {raw_response[:200]}")
    except Exception as exc:
        errors.append(f"coo_extraction_node: LLM call failed — {exc}")

    COOExtractionOutput(
        coo_extracted_fields=coo_fields,
        raw_llm_response=raw_response,
        errors=errors,
    )

    logger.info("coo_extraction_node: extracted %d fields", len(coo_fields))

    return {  # type: ignore[return-value]
        **state,
        "coo_extracted_fields": coo_fields,
        "errors": errors,
        "current_step": "coo_extraction_node",
    }
