"""
PACDStructuringNode — takes structured page extractions and assembles them
into logical documents stored in PostgreSQL + Milvus (description index).

Replaces the old pacd_indexing_node (chunk-based approach).

Algorithm
---------
1. Receive structured page extractions from vision_extraction_node
   (each page has: doc_type, is_continuation, header_fields, line_items).
2. Merge pages into logical documents using heuristic boundary detection:
   - A page with doc_type set and is_continuation=False → new document starts.
   - A page with is_continuation=True → append to previous document.
3. Store each logical document in PostgreSQL (pacd_documents + pacd_line_items).
4. Generate 3 semantic chunks per document (header / table / footer) and index
   them in the pacd_chunks Milvus collection for parallel section verification.
5. Embed line item descriptions into pacd_items for legacy HS-code fallback search.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.core.config import settings
from app.langgraph.schemas.structured_extraction import (
    LineItem,
    LogicalDocument,
    PACDStructuringOutput,
)
from app.langgraph.state import GraphState
from app.models.embedding import embed_text_chunks
from app.services import pacd_milvus_service, postgres_service

logger = logging.getLogger(__name__)


def _merge_pages_into_documents(
    structured_pages: list[dict[str, Any]],
    filename: str,
) -> list[LogicalDocument]:
    """Merge per-page structured extractions into logical documents.

    Heuristic:
    - Page with doc_type != None and is_continuation == False → new document.
    - Page with is_continuation == True → append items/headers to current doc.
    - If first page has no doc_type, create an 'other' document.
    """
    documents: list[LogicalDocument] = []
    current_doc: LogicalDocument | None = None

    for page_data in sorted(structured_pages, key=lambda p: p.get("page_num", 0)):
        page_num = page_data.get("page_num", 0)
        doc_type = page_data.get("doc_type")
        is_continuation = page_data.get("is_continuation", False)
        header_fields = page_data.get("header_fields", {})
        line_items_raw = page_data.get("line_items", [])

        # Parse line items into LineItem models
        line_items = []
        for idx, item_data in enumerate(line_items_raw, start=1):
            if isinstance(item_data, dict):
                # Assign item_number if not present
                if not item_data.get("item_number"):
                    item_data["item_number"] = idx
                try:
                    line_items.append(LineItem(**item_data))
                except Exception:
                    # Best effort — include with just description
                    line_items.append(LineItem(
                        item_number=idx,
                        description=item_data.get("description", str(item_data)),
                    ))

        # Decision: new document or continuation?
        if is_continuation and current_doc is not None:
            # Continuation — merge into current document
            current_doc.page_numbers.append(page_num)
            # Merge any new header fields (e.g. "continued" header)
            for k, v in header_fields.items():
                if k not in current_doc.header_fields:
                    current_doc.header_fields[k] = v
            # Append line items with adjusted numbering
            existing_max = max((li.item_number or 0) for li in current_doc.line_items) if current_doc.line_items else 0
            for li in line_items:
                if li.item_number and li.item_number <= existing_max:
                    li.item_number = existing_max + (line_items.index(li) + 1)
                current_doc.line_items.append(li)
        else:
            # New document starts
            if current_doc is not None:
                documents.append(current_doc)
            current_doc = LogicalDocument(
                doc_type=doc_type or "other",
                filename=filename,
                page_numbers=[page_num],
                header_fields=header_fields,
                line_items=line_items,
            )

    # Don't forget the last document
    if current_doc is not None:
        documents.append(current_doc)

    return documents


# Keywords that identify footer-level fields in header_fields
_FOOTER_KEYWORDS = frozenset({
    "total_", "remark", "term", "signature", "payment", "bank",
    "seal", "stamp", "authorized", "note", "comment", "condition",
    "balance", "due", "freight", "insurance", "grand",
})


def _is_footer_field(key: str) -> bool:
    k = key.lower()
    return any(k.startswith(kw) or kw in k for kw in _FOOTER_KEYWORDS)


def _build_chunk_records(
    logical_docs: list[LogicalDocument],
    transaction_id: str,
    document_id: str,
) -> list[dict]:
    """Build header / table / footer chunk records for every logical document.

    Returns a list of dicts ready for embedding (without dense/sparse vectors).
    Each record: id, transaction_id, document_id, doc_type,
                 chunk_type, chunk_text, chunk_metadata.
    """
    import json

    records: list[dict] = []

    for doc in logical_docs:
        doc_id_slug = f"{document_id}_{doc.doc_type}"

        # ── Header chunk ──────────────────────────────────────────────────
        header_fields = {k: v for k, v in doc.header_fields.items() if not _is_footer_field(k) and v}
        if header_fields:
            header_text = f"[{doc.doc_type}] " + " | ".join(
                f"{k}: {v}" for k, v in header_fields.items()
            )
            records.append({
                "id":             f"{doc_id_slug}_header",
                "transaction_id": transaction_id,
                "document_id":    document_id,
                "doc_type":       doc.doc_type,
                "chunk_type":     "header",
                "chunk_text":     header_text[:4096],
                "chunk_metadata": json.dumps({"header_fields": header_fields, "doc_type": doc.doc_type,
                                              "filename": doc.filename})[:4096],
            })

        # ── Table chunk ───────────────────────────────────────────────────
        if doc.line_items:
            lines: list[str] = []
            for li in doc.line_items:
                parts: list[str] = []
                if li.item_number:    parts.append(f"Item {li.item_number}")
                if li.hs_code:        parts.append(f"HS {li.hs_code}")
                if li.description:    parts.append(li.description)
                if li.quantity:       parts.append(f"qty: {li.quantity}")
                if li.unit:           parts.append(li.unit)
                if li.unit_price:     parts.append(f"price: {li.unit_price}")
                if li.total_value:    parts.append(f"total: {li.total_value}")
                if li.weight:         parts.append(f"weight: {li.weight}")
                if li.origin_country: parts.append(f"origin: {li.origin_country}")
                lines.append(" | ".join(parts))
            table_text = f"[{doc.doc_type} items]\n" + "\n".join(lines)
            items_meta = [
                {
                    "item_number":    li.item_number,
                    "hs_code":        li.hs_code,
                    "description":    li.description,
                    "quantity":       li.quantity,
                    "unit":           li.unit,
                    "unit_price":     li.unit_price,
                    "total_value":    li.total_value,
                    "weight":         li.weight,
                    "origin_country": li.origin_country,
                }
                for li in doc.line_items
            ]
            records.append({
                "id":             f"{doc_id_slug}_table",
                "transaction_id": transaction_id,
                "document_id":    document_id,
                "doc_type":       doc.doc_type,
                "chunk_type":     "table",
                "chunk_text":     table_text[:4096],
                "chunk_metadata": json.dumps({"line_items": items_meta, "doc_type": doc.doc_type,
                                              "filename": doc.filename})[:4096],
            })

        # ── Footer chunk ──────────────────────────────────────────────────
        footer_fields = {k: v for k, v in doc.header_fields.items() if _is_footer_field(k) and v}
        if footer_fields:
            footer_text = f"[{doc.doc_type} footer] " + " | ".join(
                f"{k}: {v}" for k, v in footer_fields.items()
            )
            records.append({
                "id":             f"{doc_id_slug}_footer",
                "transaction_id": transaction_id,
                "document_id":    document_id,
                "doc_type":       doc.doc_type,
                "chunk_type":     "footer",
                "chunk_text":     footer_text[:4096],
                "chunk_metadata": json.dumps({"footer_fields": footer_fields, "doc_type": doc.doc_type,
                                              "filename": doc.filename})[:4096],
            })

    return records


# ── Embedding helpers ─────────────────────────────────────────────────────────

_MIN_EMBED_LEN = 20  # texts shorter than this reliably produce NaN in BGE-M3 fp16


def _sanitize_embed_text(text: str) -> str:
    """Return a safe embedding input, padding short texts to _MIN_EMBED_LEN."""
    text = text.strip()
    if len(text) < _MIN_EMBED_LEN:
        text = text.ljust(_MIN_EMBED_LEN)
    return text


def _is_valid_dense(vec: list[float]) -> bool:
    """Return True iff every element is finite (no NaN or Inf)."""
    return all(math.isfinite(v) for v in vec)


def _embed_and_filter(
    records: list[dict],
    text_key: str,
) -> list[dict]:
    """Embed record texts, attach vectors, and drop records with NaN/Inf vectors.

    Modifies records in-place for valid ones; skips and logs invalid ones.
    Returns the list of valid records ready for Milvus upsert.
    """
    if not records:
        return []

    texts = [_sanitize_embed_text(r[text_key]) for r in records]
    dense_vecs, sparse_vecs = embed_text_chunks(texts)

    valid: list[dict] = []
    for i, record in enumerate(records):
        dv = dense_vecs[i]
        if not _is_valid_dense(dv):
            logger.warning(
                "pacd_structuring_node: skipping record '%s' — NaN/Inf in dense embedding "
                "(text=%r)", record.get("id"), texts[i][:80],
            )
            continue
        record["dense"] = dv
        record["sparse"] = sparse_vecs[i]
        valid.append(record)

    if len(valid) < len(records):
        logger.info(
            "pacd_structuring_node: %d/%d records had valid embeddings",
            len(valid), len(records),
        )
    return valid


def pacd_structuring_node(state: GraphState) -> GraphState:
    """LangGraph node: structure PACD pages into logical documents and persist.

    Reads
    -----
    state["structured_pages"]   — list of PageStructuredExtraction dicts
    state["document_id"], state["user_id"], state["transaction_id"]
    state["filename"], state["page_images"]

    Writes
    ------
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("pacd_structuring_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    structured_pages: list[dict] = state.get("structured_pages") or []
    document_id = state.get("document_id") or ""
    user_id = state.get("user_id") or ""
    transaction_id = state.get("transaction_id") or ""
    filename = state.get("filename") or "unknown"

    if not structured_pages:
        errors.append("pacd_structuring_node: no structured_pages in state")
        return {**state, "errors": errors, "current_step": "pacd_structuring_node"}  # type: ignore[return-value]

    # ── Step 1: Merge pages into logical documents ────────────────────────────
    logical_docs = _merge_pages_into_documents(structured_pages, filename)
    logger.info(
        "pacd_structuring_node: merged %d pages into %d logical documents",
        len(structured_pages), len(logical_docs),
    )

    # ── Step 2: Store in PostgreSQL ───────────────────────────────────────────
    total_items_created = 0
    all_milvus_records: list[dict[str, Any]] = []

    for doc in logical_docs:
        try:
            doc_id = postgres_service.insert_pacd_document(
                transaction_id=transaction_id,
                user_id=user_id,
                doc_type=doc.doc_type,
                filename=doc.filename,
                page_numbers=doc.page_numbers,
                header_fields=doc.header_fields,
                raw_ocr_text=doc.raw_ocr_text,
            )
        except Exception as exc:
            errors.append(f"pacd_structuring_node: failed to insert document ({doc.doc_type}): {exc}")
            continue

        # Insert line items
        if doc.line_items:
            items_data = [
                {
                    "item_number": li.item_number,
                    "hs_code": li.hs_code,
                    "description": li.description,
                    "quantity": li.quantity,
                    "unit": li.unit,
                    "unit_price": li.unit_price,
                    "total_value": li.total_value,
                    "weight": li.weight,
                    "origin_country": li.origin_country,
                    "raw_text": li.raw_text,
                }
                for li in doc.line_items
            ]
            try:
                item_ids = postgres_service.insert_pacd_line_items(
                    pacd_document_id=doc_id,
                    transaction_id=transaction_id,
                    items=items_data,
                )
                total_items_created += len(item_ids)

                # Prepare Milvus records for fuzzy description search
                for item_id, li in zip(item_ids, doc.line_items):
                    desc_text = f"{li.hs_code or ''} | {li.description}".strip(" |")
                    if desc_text:
                        all_milvus_records.append({
                            "id": item_id,
                            "transaction_id": transaction_id,
                            "hs_code": li.hs_code or "",
                            "description_text": desc_text[:2048],
                        })
            except Exception as exc:
                errors.append(f"pacd_structuring_node: failed to insert line items for {doc.doc_type}: {exc}")

    logger.info(
        "pacd_structuring_node: stored %d documents, %d line items in Postgres",
        len(logical_docs), total_items_created,
    )

    # ── Step 3: Generate semantic chunks and index in Milvus ──────────────────
    # 3a: line-item description records for legacy HS-code fallback (pacd_items)
    if all_milvus_records:
        try:
            valid_item_records = _embed_and_filter(all_milvus_records, "description_text")
            if valid_item_records:
                pacd_milvus_service.upsert_item_descriptions(valid_item_records)
                logger.info(
                    "pacd_structuring_node: indexed %d item descriptions in pacd_items",
                    len(valid_item_records),
                )
        except Exception as exc:
            errors.append(f"pacd_structuring_node: pacd_items Milvus indexing failed — {exc}")
            logger.warning("pacd_structuring_node: pacd_items Milvus error: %s", exc)

    # 3b: semantic section chunks (header / table / footer) → pacd_chunks
    chunk_records = _build_chunk_records(logical_docs, transaction_id, document_id)
    if chunk_records:
        try:
            valid_chunk_records = _embed_and_filter(chunk_records, "chunk_text")
            if valid_chunk_records:
                pacd_milvus_service.upsert_chunks(valid_chunk_records)
                logger.info(
                    "pacd_structuring_node: indexed %d/%d section chunks in pacd_chunks",
                    len(valid_chunk_records), len(chunk_records),
                )
        except Exception as exc:
            errors.append(f"pacd_structuring_node: pacd_chunks Milvus indexing failed — {exc}")
            logger.warning("pacd_structuring_node: pacd_chunks Milvus error: %s", exc)

    # Also persist page images to documents table for audit
    page_images: list[bytes] = state.get("page_images") or []
    if page_images:
        try:
            postgres_records = []
            for page_num, image_bytes in enumerate(page_images, start=1):
                postgres_records.append({
                    "id": f"{document_id}_p{page_num}",
                    "country": "",
                    "doc_type": "pacd",
                    "file_name": filename,
                    "ocr_text": "",
                    "image_data": image_bytes,
                    "image_content_type": "image/jpeg",
                })
            postgres_service.insert_documents_with_outbox(postgres_records)
        except Exception as exc:
            errors.append(f"pacd_structuring_node: page image storage failed — {exc}")

    return {  # type: ignore[return-value]
        **state,
        "errors": errors,
        "current_step": "pacd_structuring_node",
    }
