"""
VLM Chunking Node — Phase 1 of 2-Tier GraphRAG.

Converts uploaded PACD document pages into logical chunks via Vision LLM,
embeds them with BGE-M3, and stores in the ``coo_reference_chunks`` Milvus
collection.

Two chunk types are produced:
  1. Line Item chunks — one per product line, containing ALL fields for that item
  2. Document-level field chunks — one per header field (exporter, consignee, etc.)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.core.config import settings
from app.langgraph.llm import encode_image_for_llm, get_structured_llm
from app.langgraph.schemas.graphrag_schemas import (
    MAX_CHUNK_TOPIC,
    DocumentChunkList,
)
from app.langgraph.state import GraphState
from app.models.embedding import embed_text_chunks
from app.services.pacd_milvus_service import insert_reference_chunks

logger = logging.getLogger(__name__)

_CHUNKING_PROMPT = """You are a trade document field extractor. Extract all data from this trade document as a list of DocumentChunk objects using the TWO-TYPE chunking strategy below.

══════════════════════════════════════════════════════
TYPE 1 — LINE ITEM CHUNKS  (one chunk per product line)
══════════════════════════════════════════════════════
Each distinct product / goods line gets its OWN chunk containing ALL its fields.
This is critical: mixing fields from different items into one chunk is FORBIDDEN.

  chunk_topic format : "Line Item <N> — <Item Description>"
                       e.g. "Line Item 1 — Cocoa Beans"
                            "Line Item 2 — Coffee Beans"

  raw_text must contain ALL of the following visible for that item,
  formatted as "FIELD_LABEL: VALUE" pairs separated by " | ":
    • Item description / goods description   (verbatim)
    • HS Code / Tariff Heading
    • Quantity  (number + unit, e.g. "500 Sacs")
    • Net Weight  (with unit)
    • Gross Weight  (with unit)
    • Unit Price  (with currency)
    • Total / Line Value  (with currency)
    • Country of Origin
    • Item marks, grades, or codes

  Example:
    chunk_topic: "Line Item 1 — Cocoa Beans"
    raw_text:    "Description: COCOA BEANS, RAW, WHOLE, GRADE I | HS Code: 1801.00 |
                  Qty: 500 Sacs | Net Weight: 24,750.00 Kg | Gross Weight: 25,000.00 Kg |
                  Unit Price: USD 110.00/Sac | Total Value: USD 55,000.00 | Origin: Egypt"

══════════════════════════════════════════════════════
TYPE 2 — DOCUMENT-LEVEL FIELD CHUNKS  (one chunk per header field)
══════════════════════════════════════════════════════
One chunk per document header field. Do not combine multiple header fields.

  Fields to extract:  Exporter name & address, Importer/Consignee name & address,
                      Notify Party, Invoice Number & Date, Certificate Number & Issue Date,
                      Port of Loading, Port of Discharge, Bill of Lading / AWB Number,
                      Payment Terms, Incoterms, Issuing Authority.

  Example:
    chunk_topic: "Exporter"
    raw_text:    "Exporter: ACME Trading Co. Ltd, 12 Nile Street, Cairo, Egypt"

══════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════
  1. chunk_topic ≤ {max_topic} chars
  2. Copy ALL values VERBATIM — never normalize, convert units, or rephrase
  3. NEVER mix fields from two different line items into one chunk
  4. Never omit a visible field — zero data loss
  5. Do NOT invent data not on the document
  6. Do NOT add commentary outside the JSON

Extract every field exactly as it appears — do not invent data, do not omit any visible field."""


def vlm_chunking_node(state: GraphState) -> dict[str, Any]:
    """Extract logical chunks from PACD document pages via Vision LLM."""
    page_images: list[bytes] = state.get("page_images") or []
    transaction_id: str = state.get("transaction_id") or ""
    filename: str = state.get("filename") or "unknown"
    source_doc = os.path.basename(filename)

    if not page_images:
        logger.warning("vlm_chunking_node: no page images — skipping")
        return {
            "pages_indexed": 0,
            "extracted_kv_pairs": [],
            "errors": state.get("errors", []) + ["No page images available for chunking"],
            "current_step": "vlm_chunking_node",
        }

    # Build the VLM message with all pages as images
    from langchain_core.messages import HumanMessage

    prompt_text = _CHUNKING_PROMPT.format(max_topic=MAX_CHUNK_TOPIC)

    # Build multimodal content: images first, then instruction
    content: list[dict[str, Any]] = []
    for img_bytes in page_images:
        data_uri = encode_image_for_llm(img_bytes)
        content.append({"type": "image_url", "image_url": {"url": data_uri}})
    content.append({"type": "text", "text": prompt_text})

    llm = get_structured_llm(DocumentChunkList, temperature=0.0, vision=True)

    try:
        chunk_list: DocumentChunkList = llm.invoke([HumanMessage(content=content)])  # type: ignore[assignment]
    except Exception as exc:
        logger.error("vlm_chunking_node: VLM call failed — %s", exc)
        return {
            "pages_indexed": 0,
            "extracted_kv_pairs": [],
            "errors": state.get("errors", []) + [f"VLM chunking failed: {exc}"],
            "current_step": "vlm_chunking_node",
        }

    if not chunk_list.chunks:
        logger.warning("vlm_chunking_node: VLM returned 0 chunks")
        return {
            "pages_indexed": 0,
            "extracted_kv_pairs": [],
            "errors": state.get("errors", []) + ["VLM returned no chunks"],
            "current_step": "vlm_chunking_node",
        }

    # Enrich chunks with IDs
    enriched: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunk_list.chunks, 1):
        enriched.append({
            "chunk_id": f"{transaction_id}__chunk_{i:03d}",
            "transaction_id": transaction_id,
            "source_document": source_doc,
            "chunk_topic": chunk.chunk_topic,
            "raw_text": chunk.raw_text,
        })

    logger.info(
        "vlm_chunking_node: %d chunks extracted from '%s'", len(enriched), source_doc
    )

    # Embed all chunk texts with BGE-M3
    texts = [c["raw_text"] for c in enriched]
    try:
        dense_vecs, sparse_vecs = embed_text_chunks(texts)
    except Exception as exc:
        logger.error("vlm_chunking_node: embedding failed — %s", exc)
        return {
            "pages_indexed": 0,
            "extracted_kv_pairs": [],
            "errors": state.get("errors", []) + [f"Embedding failed: {exc}"],
            "current_step": "vlm_chunking_node",
        }

    # Build Milvus records
    records: list[dict[str, Any]] = []
    for i, chunk_data in enumerate(enriched):
        records.append({
            "item_id": chunk_data["chunk_id"][:128],
            "transaction_id": chunk_data["transaction_id"][:64],
            "source_document": chunk_data["source_document"][:256],
            "raw_text": chunk_data["raw_text"][:65535],
            "metadata_json": json.dumps({
                "chunk_topic": chunk_data["chunk_topic"][:MAX_CHUNK_TOPIC],
                "source_document": chunk_data["source_document"][:200],
            })[:2048],
            "dense_vector": dense_vecs[i],
            "sparse_vector": sparse_vecs[i],
        })

    # Insert into Milvus
    n_inserted = insert_reference_chunks(records)
    logger.info(
        "vlm_chunking_node: inserted %d chunks for txn='%s'",
        n_inserted, transaction_id,
    )

    # Build backward-compatible extracted_kv_pairs for API response
    kv_pairs = [
        {"page": 1, "key": c["chunk_topic"], "value": c["raw_text"][:200]}
        for c in enriched
    ]

    return {
        "pages_indexed": n_inserted,
        "extracted_kv_pairs": kv_pairs,
        "errors": state.get("errors", []),
        "current_step": "vlm_chunking_node",
    }
