"""
VisionExtractionNode — sends a document page image to the Vision LLM and
extracts all key-value pairs present on that page.

Used for:
 - PACD document pages  (doc_category = 'pacd')
 - COO document pages   (doc_category = 'coo')  with optional field_hints
 - Template images      (doc_category = 'template') to identify fillable fields
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from app.langgraph.llm import encode_image_for_llm, get_vision_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    PageKVPair,
    VisionExtractionInput,
    VisionExtractionOutput,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_GENERAL = """\
You are a precise document data extraction assistant.
Examine the provided document image and extract ALL key-value information you can see.
Include: names, dates, numbers, addresses, codes, amounts, reference numbers, and any labeled fields.

Respond ONLY with a JSON array. Each element must be:
{ "key": "<field_name_snake_case>", "value": "<extracted_value>", "confidence": <0.0-1.0> }

Rules:
- Use snake_case for key names.
- If a value is unclear, include it with lower confidence.
- Do not include empty values.
- Return an empty array [] if nothing can be extracted.
"""

_SYSTEM_PROMPT_GUIDED = """\
You are a precise document data extraction assistant.
Examine the provided document image and extract the specified fields.

Fields to extract:
{field_list}

Respond ONLY with a JSON array. Each element must be:
{{ "key": "<field_name>", "value": "<extracted_value>", "confidence": <0.0-1.0> }}

Rules:
- Use the exact field names provided above.
- Set value to null and confidence to 0 if a field is not found.
- Do not add extra fields.
"""


def vision_extraction_node(state: GraphState) -> GraphState:
    """LangGraph node: extract key-value pairs from a document page via Vision LLM.

    Reads
    -----
    state["page_images"]        — list of per-page JPEG bytes (uses first page)
    state["document_id"]
    state["doc_category"]
    state["template_attributes"] — if set, used as field_hints for guided extraction

    Writes
    ------
    state["extracted_kv_pairs"] — list of {page, key, value, confidence} dicts
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("vision_extraction_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    page_images: list[bytes] = state.get("page_images") or []
    document_id: str = state.get("document_id") or ""
    doc_category: str = state.get("doc_category") or "pacd"

    if not page_images:
        errors.append("vision_extraction_node: no page_images in state")
        return {**state, "errors": errors, "current_step": "vision_extraction_node"}  # type: ignore[return-value]

    template_attributes: dict = state.get("template_attributes") or {}
    field_hints = list(template_attributes.keys()) if template_attributes else []

    llm = get_vision_llm(temperature=0.0)
    all_kv_pairs: list[dict] = []

    for page_num, image_bytes in enumerate(page_images, start=1):
        try:
            node_input = VisionExtractionInput(
                document_id=document_id,
                page_num=page_num,
                image_b64=encode_image_for_llm(image_bytes, "image/jpeg"),
                doc_category=doc_category,  # type: ignore[arg-type]
                field_hints=field_hints,
            )
        except Exception as exc:
            errors.append(f"vision_extraction_node p{page_num} input validation: {exc}")
            continue

        if field_hints:
            system_text = _SYSTEM_PROMPT_GUIDED.format(
                field_list="\n".join(f"- {f}" for f in field_hints)
            )
        else:
            system_text = _SYSTEM_PROMPT_GENERAL

        message = HumanMessage(
            content=[
                {"type": "text", "text": system_text},
                {
                    "type": "image_url",
                    "image_url": {"url": node_input.image_b64},
                },
            ]
        )

        try:
            response = llm.invoke([message])
            raw = str(response.content)
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Expected JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"vision_extraction_node p{page_num}: bad LLM JSON — {exc}")
            continue
        except Exception as exc:
            errors.append(f"vision_extraction_node p{page_num}: LLM call failed — {exc}")
            continue

        for item in parsed:
            try:
                kv = PageKVPair(**item)
                if kv.value:
                    all_kv_pairs.append(
                        {"page": page_num, "key": kv.key, "value": kv.value, "confidence": kv.confidence}
                    )
            except Exception as exc:
                errors.append(f"vision_extraction_node p{page_num}: skipping malformed item {item}: {exc}")

    VisionExtractionOutput(
        document_id=document_id,
        page_num=1,
        kv_pairs=[PageKVPair(key=kv["key"], value=kv["value"], confidence=kv["confidence"]) for kv in all_kv_pairs],
        errors=errors,
    )

    logger.info("vision_extraction_node: extracted %d kv pairs across %d pages", len(all_kv_pairs), len(page_images))

    return {  # type: ignore[return-value]
        **state,
        "extracted_kv_pairs": all_kv_pairs,
        "errors": errors,
        "current_step": "vision_extraction_node",
    }
