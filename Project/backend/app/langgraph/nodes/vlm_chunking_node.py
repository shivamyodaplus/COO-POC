"""
VLM Chunking Node — Phase 1 of 2-Tier GraphRAG.

Converts uploaded PACD document pages into logical chunks via Vision LLM,
embeds them with BGE-M3, and stores in the ``coo_reference_chunks`` Milvus
collection.

Strategy: Each page is processed individually by the VLM which segments it into
three logical sections:
  1. Header — document-level identifiers (exporter, consignee, dates, ports, etc.)
  2. Content — body/transactional data (line items, goods, quantities, prices, etc.)
  3. Footer — stamps, signatures, issuing authority seals, certification notes

Large sections (>1000 chars) are split into multiple chunks at line boundaries.
During retrieval, chunks are sorted by page → content_type (H→C→F) for readability.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage

from app.langgraph.llm import encode_image_for_llm, get_structured_llm
from app.langgraph.schemas.graphrag_schemas import MAX_CHUNK_SIZE, PageSections
from app.langgraph.state import GraphState
from app.models.embedding import embed_text_chunks
from app.services.pacd_milvus_service import insert_reference_chunks

logger = logging.getLogger(__name__)

_CHUNKING_PROMPT = """You are a trade document page segmenter. Analyze this SINGLE page of a trade document and divide ALL visible text into exactly three sections.

══════════════════════════════════════════════════════
HEADER — Document-level identifiers (top of page)
══════════════════════════════════════════════════════
Include ALL of the following if visible on this page:
  • Exporter name & address
  • Importer / Consignee name & address
  • Notify Party
  • Invoice Number & Date
  • Certificate Number & Issue Date
  • Port of Loading / Port of Discharge
  • Bill of Lading / AWB Number
  • Payment Terms / Incoterms
  • Country of Origin (document-level declaration)
  • Any other document identifiers, reference numbers, or party details at the top

══════════════════════════════════════════════════════
CONTENT — Body / transactional data (middle of page)
══════════════════════════════════════════════════════
Include ALL of the following if visible on this page:
  • Line items / goods descriptions
  • HS Codes / Tariff Headings
  • Quantities (number + unit)
  • Net Weight / Gross Weight
  • Unit Prices / Total Values
  • Packing details, marks, grades
  • Totals, subtotals, summaries
  • Any tabular or list data describing the goods

══════════════════════════════════════════════════════
FOOTER — Authentication marks (bottom of page)
══════════════════════════════════════════════════════
Include ALL of the following if visible on this page:
  • Stamps (official, customs, chamber of commerce)
  • Signatures (authorized signatory, representative)
  • Issuing authority seals
  • Certification notes / authentication marks
  • Date and place of issue (if at the bottom)

══════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════
  1. Copy ALL text VERBATIM — never normalize, convert, summarize, or rephrase
  2. Never omit any visible text — zero data loss
  3. If a section has no content on this page, return an empty string for it
  4. Do NOT invent data not on the document
  5. Do NOT add commentary outside the JSON
  6. Preserve line breaks and formatting as closely as possible

Extract every field exactly as it appears on this page."""


def _split_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split text into chunks of approximately max_size at line boundaries."""
    if not text or len(text) <= max_size:
        return [text] if text else []

    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current and (current_len + line_len) > max_size:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def vlm_chunking_node(state: GraphState) -> dict[str, Any]:
    """Extract logical chunks from PACD document pages via Vision LLM.

    Processes each page individually, segments into Header/Content/Footer,
    splits large sections, embeds with BGE-M3, and stores in Milvus.
    """
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

    llm = get_structured_llm(PageSections, temperature=0.0, vision=True)
    all_chunks: list[dict[str, Any]] = []
    errors: list[str] = list(state.get("errors", []))

    # Process each page individually
    for page_num, img_bytes in enumerate(page_images, 1):
        data_uri = encode_image_for_llm(img_bytes)
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": _CHUNKING_PROMPT},
        ]

        try:
            page_sections: PageSections = llm.invoke([HumanMessage(content=content)])  # type: ignore[assignment]
        except Exception as exc:
            logger.error(
                "vlm_chunking_node: VLM failed on page %d — %s", page_num, exc
            )
            errors.append(f"VLM chunking failed on page {page_num}: {exc}")
            continue

        # Build chunks for each non-empty section
        for content_type in ("header", "content", "footer"):
            section_text: str = getattr(page_sections, content_type, "").strip()
            if not section_text:
                continue

            # Split large sections into multiple chunks
            text_parts = _split_text(section_text)
            for sub_idx, part in enumerate(text_parts, 1):
                chunk_id = (
                    f"{transaction_id}__p{page_num:02d}_{content_type}_{sub_idx:02d}"
                )
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "transaction_id": transaction_id,
                    "source_document": source_doc,
                    "page_number": page_num,
                    "content_type": content_type,
                    "chunk_index": sub_idx,
                    "raw_text": part,
                })

    if not all_chunks:
        logger.warning("vlm_chunking_node: no chunks produced from %d pages", len(page_images))
        return {
            "pages_indexed": 0,
            "extracted_kv_pairs": [],
            "errors": errors + ["VLM returned no content from any page"],
            "current_step": "vlm_chunking_node",
        }

    logger.info(
        "vlm_chunking_node: %d chunks from %d pages of '%s'",
        len(all_chunks), len(page_images), source_doc,
    )

    # Embed all chunk texts with BGE-M3
    texts = [c["raw_text"] for c in all_chunks]
    try:
        dense_vecs, sparse_vecs = embed_text_chunks(texts)
    except Exception as exc:
        logger.error("vlm_chunking_node: embedding failed — %s", exc)
        return {
            "pages_indexed": 0,
            "extracted_kv_pairs": [],
            "errors": errors + [f"Embedding failed: {exc}"],
            "current_step": "vlm_chunking_node",
        }

    # Build Milvus records
    records: list[dict[str, Any]] = []
    for i, chunk_data in enumerate(all_chunks):
        records.append({
            "item_id": chunk_data["chunk_id"][:128],
            "transaction_id": chunk_data["transaction_id"][:64],
            "source_document": chunk_data["source_document"][:256],
            "raw_text": chunk_data["raw_text"][:65535],
            "metadata_json": json.dumps({
                "page_number": chunk_data["page_number"],
                "content_type": chunk_data["content_type"],
                "chunk_index": chunk_data["chunk_index"],
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
        {
            "page": c["page_number"],
            "key": f"{c['content_type']} (p{c['page_number']})",
            "value": c["raw_text"][:200],
        }
        for c in all_chunks
    ]

    return {
        "pages_indexed": n_inserted,
        "extracted_kv_pairs": kv_pairs,
        "errors": errors,
        "current_step": "vlm_chunking_node",
    }
