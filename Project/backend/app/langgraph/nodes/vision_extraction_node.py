"""
VisionExtractionNode — sends a document page image to the Vision LLM and
extracts structured data from that page.

For PACD documents: Uses ``with_structured_output(PageStructuredExtraction)``
to directly output header fields + line items per page (no JSON parsing needed).

For COO / template documents: Falls back to general KV extraction (flat pairs).
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
from app.langgraph.schemas.structured_extraction import PageStructuredExtraction
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

_SYSTEM_PROMPT_STRUCTURED_PACD = """\
You are a precise document data extraction assistant specializing in trade/shipping documents.

Examine the provided document image and extract structured data.

Identify:
1. **Document type** — what kind of document is this page? (commercial_invoice, packing_list, bill_of_lading, certificate_of_origin, insurance_certificate, contract, other). Set to null if this is a continuation page.
2. **Is this a continuation?** — True if this page continues a document from the previous page (e.g. more line items, no new header).
3. **Header fields** — metadata/header information (exporter, consignee, invoice number, date, port, vessel, etc.). Use snake_case keys.
4. **Line items** — if the page contains a table of products/goods, extract each row as a separate item with: hs_code, description, quantity, unit, unit_price, total_value, weight, origin_country. Only include fields that are actually present.

Important rules:
- Header fields are document-level metadata (appears once, at the top or in a header area).
- Line items are repeated/tabular data (products, goods, items in a list or table).
- If no line items exist (e.g. a bill of lading with no product table), leave line_items empty.
- If no header is visible (continuation page), set doc_type to null and is_continuation to true.
"""


def _extract_structured_pacd_page(
    image_bytes: bytes, page_num: int
) -> PageStructuredExtraction | None:
    """Use with_structured_output to extract structured PACD page data."""
    llm = get_structured_llm(
        PageStructuredExtraction, temperature=0.0, vision=True
    )
    image_b64 = encode_image_for_llm(image_bytes, "image/jpeg")
    message = HumanMessage(
        content=[
            {"type": "text", "text": _SYSTEM_PROMPT_STRUCTURED_PACD},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]
    )
    result = llm.invoke([message])
    return result  # type: ignore[return-value]


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

    # ── PACD path: structured extraction ──────────────────────────────────────
    if doc_category == "pacd":
        structured_pages: list[dict] = []
        for page_num, image_bytes in enumerate(page_images, start=1):
            try:
                result = _extract_structured_pacd_page(image_bytes, page_num)
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
