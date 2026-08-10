"""
COO Transcription Node — Phase 2a of 2-Tier GraphRAG.

Uses Vision LLM to:
  Step 1: Transcribe all visible text from the COO page images
  Step 2: Generate up to MAX_VALIDATION_QUERIES targeted verification queries
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.langgraph.llm import encode_image_for_llm, get_structured_llm, get_vision_llm
from app.langgraph.schemas.graphrag_schemas import ValidationQueryList
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)


def coo_transcription_node(state: GraphState) -> dict[str, Any]:
    """Transcribe COO text and generate validation queries via Vision LLM."""
    page_images: list[bytes] = state.get("page_images") or []
    transaction_id: str = state.get("transaction_id") or ""

    if not page_images:
        logger.warning("coo_transcription_node: no page images")
        return {
            "coo_text": "",
            "validation_queries": [],
            "errors": state.get("errors", []) + ["No COO page images available"],
            "current_step": "coo_transcription_node",
        }

    # Encode all pages as base64 data URIs
    image_contents: list[dict[str, Any]] = []
    for img_bytes in page_images:
        data_uri = encode_image_for_llm(img_bytes)
        image_contents.append({"type": "image_url", "image_url": {"url": data_uri}})

    llm = get_vision_llm(temperature=0.0)

    # ── Step 1: Transcribe COO text ──────────────────────────────────────────
    transcription_content = list(image_contents) + [{
        "type": "text",
        "text": (
            "You are a document transcription assistant. "
            "Transcribe ALL text from this Certificate of Origin document exactly as it appears — "
            "including field labels, values, dates, codes, and signatures. "
            "IMPORTANT — for tabular or columnar layouts: explicitly pair each column header with "
            "its cell value on the same row, formatted as 'COLUMN_HEADER: VALUE'. "
            "Do not emit a bare number or value without its corresponding column label. "
            "Preserve the original structure using newlines. Return only the raw transcription, no commentary."
        ),
    }]

    try:
        response = llm.invoke([HumanMessage(content=transcription_content)])
        coo_text = response.content.strip()
    except Exception as exc:
        logger.error("coo_transcription_node: transcription failed — %s", exc)
        return {
            "coo_text": "",
            "validation_queries": [],
            "errors": state.get("errors", []) + [f"COO transcription failed: {exc}"],
            "current_step": "coo_transcription_node",
        }

    logger.info("coo_transcription_node: transcribed %d chars", len(coo_text))

    # ── Step 2: Generate validation queries ──────────────────────────────────
    max_queries = settings.MAX_VALIDATION_QUERIES

    query_prompt = """You are a trade compliance auditor. Examine this Certificate of Origin (COO) document.

        Generate all search strings that will be used to retrieve matching chunks from the PACD reference documents (commercial invoice, packing list, bill of lading) stored in a vector database.

        RULES FOR EACH SEARCH STRING:
        1. Include the ACTUAL VALUE read from the COO — not the field label or a question.
            GOOD: "<company name> <address> consignee buyer"
            BAD:  "What is the consignee name and address?"
        2. Append 1-3 generic trade synonyms after the value to bridge terminology gaps
            between the COO and PACD documents (e.g. a buyer may be labelled "CLIENT" on an invoice).
            Common synonym groups:
            exporter / seller / shipper / supplier
            consignee / buyer / client / importer / livraison / delivery
            port of loading / departure port / shipped from
            port of discharge / destination port / delivered to
            HS code / tariff heading / customs code
            invoice number / bill number / reference number
            invoice date / bill date / document date
            gross weight / total weight
        3. For weight fields: emit the weight value with its label exactly as printed on the
            COO (e.g. the label the document itself uses). Append all common weight synonyms:
            net weight / gross weight / total weight / weight / quantity weight.
        4. Cover EVERY distinct field visible on the COO: exporter, consignee,
            ports of loading and discharge, each line item (HS code, goods description,
            quantity, weight, unit price, total value), invoice number, invoice date,
            country of origin.
        5. One field per search string — keep each string under 100 characters.

        Respond ONLY with this exact JSON structure — a single object with a "queries" key containing a list of strings:
        {{"queries": ["search string 1", "search string 2", "search string 3"]}}"""

    query_content = list(image_contents) + [{"type": "text", "text": query_prompt}]

    structured_llm = get_structured_llm(ValidationQueryList, temperature=0.0, vision=True)

    try:
        query_list: ValidationQueryList = structured_llm.invoke([HumanMessage(content=query_content)])  # type: ignore[assignment]
        validation_queries = query_list.queries[:max_queries]
    except Exception as exc:
        logger.warning("coo_transcription_node: query generation failed — %s", exc)
        # Fallback: extract basic queries from transcribed text
        validation_queries = _fallback_queries(coo_text)

    logger.info(
        "coo_transcription_node: generated %d validation queries", len(validation_queries)
    )

    return {
        "coo_text": coo_text,
        "validation_queries": validation_queries,
        "errors": state.get("errors", []),
        "current_step": "coo_transcription_node",
    }


def _fallback_queries(coo_text: str) -> list[str]:
    """Generate basic queries from COO text when VLM query generation fails."""
    queries = []
    # Extract lines that look like field values
    for line in coo_text.split("\n"):
        line = line.strip()
        if ":" in line and len(line) > 10 and len(line) < 200:
            key = line.split(":", 1)[0].strip()
            if key and len(key) < 60:
                queries.append(f"What is the {key}?")
        if len(queries) >= 10:
            break
    return queries
