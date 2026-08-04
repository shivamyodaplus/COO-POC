"""
TemplateAttributeNode — Vision LLM analyses a template image and returns a
mapping of all labeled / fillable fields found on the template.

This runs once when a template is uploaded.  The output is persisted back to
the ``visual_templates.extracted_attributes`` column so COO extraction can use
it as a field schema without re-running the LLM every time.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.langgraph.llm import encode_image_for_llm, get_vision_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    TemplateAttributeInput,
    TemplateAttributeOutput,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a document template analyst.
Examine the provided template image and identify ALL labeled fields, boxes, or sections
that are meant to be filled in or printed on the final document.

For each field provide:
- key: field name in snake_case (e.g. "exporter_name", "invoice_number")
- value: the label exactly as seen on the template (e.g. "Exporter Name", "Invoice No.")

Return an empty dict if no labeled fields are found.
"""


class _AttributeMap(BaseModel):
    """Structured output schema for template attribute extraction."""

    attributes: dict[str, str]


def template_attribute_node(state: GraphState) -> GraphState:
    """LangGraph node: extract template field schema via Vision LLM.

    Reads
    -----
    state["page_images"]   — first image is used as the template image
    state["document_id"]   — used as template_id proxy
    state["filename"]      — used as template_name proxy

    Writes
    ------
    state["template_attributes"]  — {field_key: label} dict
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("template_attribute_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    page_images: list[bytes] = state.get("page_images") or []
    template_id = state.get("document_id") or ""
    template_name = state.get("filename") or "template"

    if not page_images:
        errors.append("template_attribute_node: no page_images in state")
        return {**state, "errors": errors, "current_step": "template_attribute_node"}  # type: ignore[return-value]

    image_bytes = page_images[0]

    try:
        node_input = TemplateAttributeInput(
            template_id=template_id,
            template_name=template_name,
            image_b64=encode_image_for_llm(image_bytes, "image/jpeg"),
        )
    except Exception as exc:
        errors.append(f"template_attribute_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "template_attribute_node"}  # type: ignore[return-value]

    # with_structured_output uses tool-calling to guarantee valid JSON that
    # matches _AttributeMap — no markdown fences, no truncation surprises.
    llm = get_vision_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(_AttributeMap)
    message = HumanMessage(
        content=[
            {"type": "text", "text": _SYSTEM_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": node_input.image_b64},
            },
        ]
    )

    attributes: dict[str, str] = {}
    try:
        result: _AttributeMap = structured_llm.invoke([message])  # type: ignore[assignment]
        attributes = {str(k): str(v) for k, v in result.attributes.items()}
    except Exception as exc:
        errors.append(f"template_attribute_node: LLM call failed — {exc}")

    output = TemplateAttributeOutput(
        template_id=template_id,
        attributes=attributes,
        raw_llm_response="",
        errors=errors,
    )

    logger.info("template_attribute_node: found %d attributes", len(output.attributes))

    return {  # type: ignore[return-value]
        **state,
        "template_attributes": output.attributes,
        "errors": output.errors,
        "current_step": "template_attribute_node",
    }
