"""
COOExtractionNode — Vision LLM extracts structured field values from the COO
document using ``with_structured_output(COOStructuredExtraction)``.

Outputs header_fields (dict) + line_items (list) for downstream cross-reference.
If a template is confirmed, uses the template's field schema as guidance.
Falls back to general structured extraction if no template is available.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from app.langgraph.llm import encode_image_for_llm, get_structured_llm
from app.langgraph.schemas.structured_extraction import COOLineItem, COOStructuredExtraction
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_GUIDED = """\
You are a precise document data extraction assistant specializing in Certificates of Origin (COO).

Extract structured data from the provided COO document image.

The document uses this template with the following labeled fields:
{field_list}

Extract:
1. **header_fields** — all header/metadata fields (use the exact field names listed above as keys).
2. **line_items** — each product/goods entry listed on the certificate. For each item extract:
   hs_code, description, quantity, unit, value, weight, origin_country (only include what's present).

Rules:
- Use snake_case for header field keys.
- If a field is not visible or blank, omit it entirely (do not set null).
- CRITICAL: If a field contains only its own label/placeholder text (e.g. "Importing country",
  "Insert name here", "N/A", "...") rather than an actual filled-in value, omit it entirely.
- Each distinct product with its own line/row should be a separate line_item entry.
- If items are numbered on the document, preserve the item_number.
"""

_SYSTEM_PROMPT_GENERAL = """\
You are a precise document data extraction assistant specializing in Certificates of Origin (COO).

Extract structured data from the provided COO document image.

Extract:
1. **header_fields** — all header/metadata fields as a dictionary with snake_case keys.
   Expected fields include (extract any you find): exporter_name, consignee_name,
   country_of_origin, transport_details, port_of_loading, port_of_discharge,
   certificate_number, date_of_issue, issuing_authority, invoice_number, remarks.
2. **line_items** — each product/goods entry listed on the certificate. For each item extract:
   hs_code, description, quantity, unit, value, weight, origin_country (only include what's present).

Rules:
- If a field is not visible or blank, omit it entirely.
- CRITICAL: If a field contains only its own label/placeholder text (e.g. "Importing country",
  "Insert name here", "N/A", "...") rather than an actual filled-in value, omit it entirely.
- Each distinct product should be a separate line_item entry.
- If items are numbered on the document, preserve the item_number.
"""


def coo_extraction_node(state: GraphState) -> GraphState:
    """LangGraph node: extract structured COO data via Vision LLM.

    Uses with_structured_output(COOStructuredExtraction) for reliable parsing.

    Reads
    -----
    state["page_images"]         — COO page images
    state["template_attributes"] — field schema from confirmed template
    state["document_id"]

    Writes
    ------
    state["coo_extracted_fields"]  — {field_key: value} flat dict (backward compat)
    state["coo_extracted_data"]    — {"header_fields": {...}, "line_items": [...]}
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

    # Build prompt based on template availability
    if template_attributes:
        field_list = "\n".join(f"- {k}: {v}" for k, v in template_attributes.items())
        system_text = _SYSTEM_PROMPT_GUIDED.format(field_list=field_list)
    else:
        system_text = _SYSTEM_PROMPT_GENERAL

    # Use structured output — guaranteed to return COOStructuredExtraction
    llm = get_structured_llm(COOStructuredExtraction, temperature=0.0, vision=True)

    # Process all pages and merge results
    merged_headers: dict[str, str] = {}
    merged_items: list[dict] = []

    for page_num, image_bytes in enumerate(page_images, start=1):
        image_b64 = encode_image_for_llm(image_bytes, "image/jpeg")
        message = HumanMessage(
            content=[
                {"type": "text", "text": system_text},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ]
        )

        try:
            result: COOStructuredExtraction = llm.invoke([message])  # type: ignore[assignment]

            # Merge header fields (first page takes priority)
            for k, v in result.header_fields.items():
                if k not in merged_headers:
                    merged_headers[k] = v

            # Append line items with page tracking
            for item in result.line_items:
                item_dict = item.model_dump(exclude_none=True)
                item_dict["_source_page"] = page_num
                merged_items.append(item_dict)

            logger.info(
                "coo_extraction_node p%d: %d headers, %d items",
                page_num, len(result.header_fields), len(result.line_items),
            )
        except Exception as exc:
            errors.append(f"coo_extraction_node p{page_num}: extraction failed — {exc}")
            logger.warning("coo_extraction_node p%d: %s", page_num, exc)

    # Build structured output
    coo_extracted_data = {
        "header_fields": merged_headers,
        "line_items": merged_items,
    }

    # Also produce flat dict for backward compatibility
    coo_extracted_fields = dict(merged_headers)
    for i, item in enumerate(merged_items, start=1):
        prefix = f"item_{item.get('item_number', i)}"
        for field in ["hs_code", "description", "quantity", "value", "weight"]:
            if field in item:
                coo_extracted_fields[f"{prefix}_{field}"] = item[field]

    logger.info(
        "coo_extraction_node: extracted %d header fields + %d line items",
        len(merged_headers), len(merged_items),
    )

    return {  # type: ignore[return-value]
        **state,
        "coo_extracted_fields": coo_extracted_fields,
        "coo_extracted_data": coo_extracted_data,
        "errors": errors,
        "current_step": "coo_extraction_node",
    }
