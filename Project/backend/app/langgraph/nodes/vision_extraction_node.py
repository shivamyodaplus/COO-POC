"""
VisionExtractionNode — sends a document page image to the Vision LLM and
extracts structured data from that page.

For PACD documents: two-stage extraction —
  1. Vision LLM outputs raw JSON (no tool schema / constrained decoding overhead).
  2. JSON is parsed locally into PageStructuredExtraction via Pydantic.
  If parsing fails, a text-only LLM fallback handles structuring.

For COO / template documents: general KV extraction (flat pairs).
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import encode_image_for_llm, get_structured_llm, get_vision_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    PageKVPair,
    VisionExtractionInput,
    VisionExtractionOutput,
)
from app.langgraph.schemas.structured_extraction import LineItem, PageStructuredExtraction
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

_SYSTEM_PROMPT_PACD_RAW = """\
You are a trade document data extraction assistant.
Examine the document image and respond ONLY with a single JSON object — no explanation, no markdown.

JSON shape:
{
  "doc_type": "commercial_invoice" | "packing_list" | "bill_of_lading" | "certificate_of_origin" | "insurance_certificate" | "contract" | "other" | null,
  "is_continuation": false,
  "header_fields": {"snake_case_key": "value"},
  "line_items": [
    {"description": "...", "hs_code": "...", "quantity": "...", "unit": "...", "unit_price": "...", "total_value": "...", "weight": "...", "origin_country": "..."}
  ]
}

Rules:
- doc_type: set to null if this is a continuation page with no new header.
- is_continuation: true when this page continues the previous page (more rows, no new header).
- header_fields: document-level metadata only (exporter, consignee, invoice_number, date, port, vessel …).
- line_items: one object per product row; omit fields not present in the document.
- If no line items exist, use an empty array.
"""


def _extract_pacd_page_raw(
    llm, image_bytes: bytes, page_num: int
) -> PageStructuredExtraction | None:
    """Stage 1: call Vision LLM without structured output — returns raw JSON string.
    Stage 2: parse locally via Pydantic.  Falls back to text LLM if parsing fails.
    """
    image_b64 = encode_image_for_llm(image_bytes, "image/jpeg")
    message = HumanMessage(
        content=[
            {"type": "text", "text": _SYSTEM_PROMPT_PACD_RAW},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]
    )
    response = llm.invoke([message])
    raw = str(response.content).strip()
    result = _parse_raw_json(raw, page_num)
    if result is None:
        result = _text_llm_fallback(raw, page_num)
    return result


def _parse_raw_json(raw: str, page_num: int) -> PageStructuredExtraction | None:
    """Strip LLM artifacts then construct PageStructuredExtraction from the JSON."""
    # Remove <think>...</think> blocks
    if "<think>" in raw:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Extract first JSON object if there is surrounding prose
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            raw = m.group(0).strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")

        items: list[LineItem] = []
        for i, row in enumerate(data.get("line_items") or [], start=1):
            if not isinstance(row, dict):
                continue
            row.setdefault("item_number", i)
            # Drop null/empty values so Pydantic optional fields stay None
            cleaned = {k: v for k, v in row.items() if v is not None and v != ""}
            if not cleaned.get("description"):
                continue
            try:
                items.append(LineItem(**cleaned))
            except Exception:
                logger.debug("_parse_raw_json p%d: skipping malformed item %d", page_num, i)

        return PageStructuredExtraction(
            doc_type=data.get("doc_type") or None,
            is_continuation=bool(data.get("is_continuation", False)),
            header_fields={
                str(k): str(v)
                for k, v in (data.get("header_fields") or {}).items()
                if v is not None and v != ""
            },
            line_items=items,
        )
    except Exception as exc:
        logger.warning("_parse_raw_json p%d: failed (%s)", page_num, exc)
        return None


def _text_llm_fallback(raw_text: str, page_num: int) -> PageStructuredExtraction | None:
    """Use text-only LLM with structured output to parse what the vision LLM returned.

    No image tokens involved, so the schema fits comfortably within the text
    model's context window.
    """
    llm = get_structured_llm(PageStructuredExtraction, temperature=0.0, vision=False)
    prompt = (
        "Parse the following text extracted from a trade/shipping document page "
        "into the required structured format.\n\n"
        f"Extracted text:\n{raw_text[:4000]}"
    )
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        logger.info("_text_llm_fallback p%d: structured output succeeded", page_num)
        return result  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("_text_llm_fallback p%d: also failed — %s", page_num, exc)
        return None


def vision_extraction_node(state: GraphState) -> GraphState:
    """LangGraph node: extract key-value pairs from a document page via Vision LLM.

    For PACD documents (doc_category='pacd'), uses structured output to get
    header_fields + line_items per page. The structured data is stored in
    state["structured_pages"] for downstream use by pacd_structuring_node.

    For other doc types, produces flat KV pairs in state["extracted_kv_pairs"].

    Reads
    -----
    state["page_images"]        — list of per-page JPEG bytes
    state["document_id"]
    state["doc_category"]
    state["template_attributes"] — if set, used as field_hints for guided extraction

    Writes
    ------
    state["extracted_kv_pairs"]  — list of {page, key, value, confidence} dicts (non-PACD)
    state["structured_pages"]    — list of PageStructuredExtraction dicts (PACD)
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

    # ── PACD path: two-stage extraction ────────────────────────────────────────────
    if doc_category == "pacd":
        vision_llm = get_vision_llm(temperature=0.0)
        structured_pages: list[dict] = []
        for page_num, image_bytes in enumerate(page_images, start=1):
            try:
                result = _extract_pacd_page_raw(vision_llm, image_bytes, page_num)
                if result is not None:
                    page_data = result.model_dump()
                    page_data["page_num"] = page_num
                    structured_pages.append(page_data)
                    logger.info(
                        "vision_extraction_node p%d: doc_type=%s, headers=%d, items=%d",
                        page_num,
                        page_data.get("doc_type"),
                        len(page_data.get("header_fields", {})),
                        len(page_data.get("line_items", [])),
                    )
            except Exception as exc:
                errors.append(f"vision_extraction_node p{page_num}: structured extraction failed — {exc}")
                logger.warning("vision_extraction_node p%d: %s", page_num, exc)

        # Also produce flat KV pairs for backward compatibility / audit
        all_kv_pairs: list[dict] = []
        for page_data in structured_pages:
            pn = page_data["page_num"]
            for k, v in page_data.get("header_fields", {}).items():
                all_kv_pairs.append({"page": pn, "key": k, "value": v, "confidence": 0.9})
            for item in page_data.get("line_items", []):
                desc = item.get("description", "")
                hs = item.get("hs_code", "")
                if desc:
                    all_kv_pairs.append({"page": pn, "key": f"item_{item.get('item_number', '?')}_description", "value": desc, "confidence": 0.9})
                if hs:
                    all_kv_pairs.append({"page": pn, "key": f"item_{item.get('item_number', '?')}_hs_code", "value": hs, "confidence": 0.9})

        logger.info(
            "vision_extraction_node: structured extraction complete — %d pages, %d total kv pairs",
            len(structured_pages), len(all_kv_pairs),
        )
        return {  # type: ignore[return-value]
            **state,
            "extracted_kv_pairs": all_kv_pairs,
            "structured_pages": structured_pages,
            "errors": errors,
            "current_step": "vision_extraction_node",
        }

    # ── Non-PACD path: flat KV extraction (COO, template) ────────────────────
    template_attributes: dict = state.get("template_attributes") or {}
    field_hints = list(template_attributes.keys()) if template_attributes else []

    llm = get_vision_llm(temperature=0.0)
    all_kv_pairs_flat: list[dict] = []

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
            raw = str(response.content).strip()
            if "<think>" in raw:
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if not raw.startswith("["):
                m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
                if m:
                    raw = m.group(0).strip()
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
                    all_kv_pairs_flat.append(
                        {"page": page_num, "key": kv.key, "value": kv.value, "confidence": kv.confidence}
                    )
            except Exception as exc:
                errors.append(f"vision_extraction_node p{page_num}: skipping malformed item {item}: {exc}")

    VisionExtractionOutput(
        document_id=document_id,
        page_num=1,
        kv_pairs=[PageKVPair(key=kv["key"], value=kv["value"], confidence=kv["confidence"]) for kv in all_kv_pairs_flat],
        errors=errors,
    )

    logger.info("vision_extraction_node: extracted %d kv pairs across %d pages", len(all_kv_pairs_flat), len(page_images))

    return {  # type: ignore[return-value]
        **state,
        "extracted_kv_pairs": all_kv_pairs_flat,
        "errors": errors,
        "current_step": "vision_extraction_node",
    }
