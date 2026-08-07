"""
PACDLayoutChunkingNode — Map phase of the Map-Reduce RAG pipeline.

Replaces ``pacd_structuring_node`` with a layout-aware, granular chunking
strategy designed for large PACD documents (up to 50 pages, 500+ line items).

Key differences from the old node
----------------------------------
1. **Granular table splitting**: large tables are split into chunks of
   ``MAX_TABLE_ITEMS_PER_CHUNK`` items (default 20) instead of one monolithic
   chunk.  This keeps every table chunk within the embedding token limit and
   prevents the "lost in the middle" problem during LLM verification.
2. **Markdown table format**: items are serialised as pipe-delimited markdown
   tables so the LLM critic receives structured, column-aligned data.
3. **Rich metadata**: every chunk carries ``hs_codes`` (unique HS codes in that
   chunk), ``extracted_keys`` (short product name tokens), and ``table_row_count``
   in ``chunk_metadata`` — used for deterministic HS-code filtering in the
   Reduce phase.
4. **New collection**: writes to ``pacd_document_chunks`` (not the deprecated
   ``pacd_chunks``), with additional fields: ``file_id``, ``page_numbers``,
   ``chunk_index``.

Algorithm
---------
1. Merge per-page ``structured_pages`` into logical documents (same heuristic
   as old node — ``doc_type + is_continuation``).
2. Store each logical document in PostgreSQL (unchanged).
3. For each logical document, build:
   a. 1 header chunk   — all non-footer header_fields
   b. N table chunks   — line items split into groups of MAX_TABLE_ITEMS_PER_CHUNK
   c. 1 footer chunk   — footer-keyword header_fields
4. Embed all chunks with BGE-M3, drop NaN/Inf vectors, upsert to Milvus.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

# Matches standard HS code formats: "8704.21", "87.04", "8704.21.00"
_HS_CODE_RE = re.compile(r'\b\d{2,4}\.\d{2,4}(?:\.\d{2,4})?\b')

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

# ── Footer field detection ─────────────────────────────────────────────────────
_FOOTER_KEYWORDS = frozenset({
    "total_", "remark", "term", "signature", "payment", "bank",
    "seal", "stamp", "authorized", "note", "comment", "condition",
    "balance", "due", "freight", "insurance", "grand",
})


def _is_footer_field(key: str) -> bool:
    k = key.lower()
    return any(k.startswith(kw) or kw in k for kw in _FOOTER_KEYWORDS)


# ── Page merging (identical logic to pacd_structuring_node) ───────────────────

def _merge_pages_into_documents(
    structured_pages: list[dict[str, Any]],
    filename: str,
) -> list[LogicalDocument]:
    """Merge per-page structured extractions into logical documents."""
    documents: list[LogicalDocument] = []
    current_doc: LogicalDocument | None = None

    for page_data in sorted(structured_pages, key=lambda p: p.get("page_num", 0)):
        page_num = page_data.get("page_num", 0)
        doc_type = page_data.get("doc_type")
        is_continuation = page_data.get("is_continuation", False)
        header_fields = page_data.get("header_fields", {})
        line_items_raw = page_data.get("line_items", [])

        line_items: list[LineItem] = []
        for idx, item_data in enumerate(line_items_raw, start=1):
            if isinstance(item_data, dict):
                if not item_data.get("item_number"):
                    item_data["item_number"] = idx
                try:
                    line_items.append(LineItem(**item_data))
                except Exception:
                    line_items.append(LineItem(
                        item_number=idx,
                        description=item_data.get("description", str(item_data)),
                    ))

        if is_continuation and current_doc is not None:
            current_doc.page_numbers.append(page_num)
            for k, v in header_fields.items():
                if k not in current_doc.header_fields:
                    current_doc.header_fields[k] = v
            existing_max = max((li.item_number or 0) for li in current_doc.line_items) if current_doc.line_items else 0
            for li in line_items:
                if li.item_number and li.item_number <= existing_max:
                    li.item_number = existing_max + (line_items.index(li) + 1)
                current_doc.line_items.append(li)
        else:
            if current_doc is not None:
                documents.append(current_doc)
            current_doc = LogicalDocument(
                doc_type=doc_type or "other",
                filename=filename,
                page_numbers=[page_num],
                header_fields=header_fields,
                line_items=line_items,
            )

    if current_doc is not None:
        documents.append(current_doc)
    return documents


# ── Markdown table builder ─────────────────────────────────────────────────────

_TABLE_HEADER = "| # | HS Code | Description | Qty | Unit | Unit Price | Total Value | Weight | Origin |"
_TABLE_SEP    = "|---|---------|-------------|-----|------|------------|-------------|--------|--------|"


def _items_to_markdown_table(items: list[LineItem]) -> str:
    rows: list[str] = [_TABLE_HEADER, _TABLE_SEP]
    for li in items:
        # Normalise the HS code so "870421" displays as "8704.21" in the table
        hs_display = _normalize_hs_code(li.hs_code) if li.hs_code else ""
        rows.append(
            f"| {li.item_number or ''} "
            f"| {hs_display} "
            f"| {li.description or ''} "
            f"| {li.quantity or ''} "
            f"| {li.unit or ''} "
            f"| {li.unit_price or ''} "
            f"| {li.total_value or ''} "
            f"| {li.weight or ''} "
            f"| {li.origin_country or ''} |"
        )
    return "\n".join(rows)


def _normalize_hs_code(raw: str) -> str:
    """Normalise a raw HS code string.

    Handles the common invoice convention where the Reference/Article column
    stores the HS code without its decimal separator:
      "870421"  → "8704.21"   (6-digit concatenated)
      "8704.21" → "8704.21"   (already correct)
      "87042100" → "8704.21"  (8-digit, trim trailing zeros)
    """
    raw = raw.strip().replace(' ', '').replace('-', '')
    # Already has a decimal — return as-is
    if '.' in raw:
        return raw
    # 6-digit code: insert dot after position 4
    if raw.isdigit() and len(raw) == 6:
        return f"{raw[:4]}.{raw[4:]}"
    # 8-digit code: insert dot after 4, ignore trailing zeros
    if raw.isdigit() and len(raw) == 8:
        return f"{raw[:4]}.{raw[4:6]}"
    return raw


def _extract_chunk_keys(items: list[LineItem]) -> tuple[list[str], list[str]]:
    """Return (hs_codes, extracted_keys) for a batch of line items.

    hs_codes are collected from three sources (in order):
    1. The dedicated ``li.hs_code`` field (normalised to dotted format).
    2. HS code patterns embedded in ``li.description`` (e.g. "HS Code: 8704.21").
    3. The ``li.raw_text`` field as a last-resort scan.

    This handles invoices where the Vision LLM extracts the HS code from a
    ``Reference`` column without decimal separator ("870421" instead of
    "8704.21"), or where the HS code only appears inline in the description.
    """
    hs_seen: set[str] = set()
    hs_codes: list[str] = []

    def _add_hs(code: str) -> None:
        normalised = _normalize_hs_code(code)
        if normalised and normalised not in hs_seen:
            hs_seen.add(normalised)
            hs_codes.append(normalised)
            # Also add the raw form so both formats are searchable
            if code != normalised and code not in hs_seen:
                hs_seen.add(code)
                hs_codes.append(code)

    for li in items:
        if li.hs_code:
            _add_hs(li.hs_code)
        # Scan description and raw_text for embedded HS code patterns
        for text_field in (li.description or '', li.raw_text or ''):
            for match in _HS_CODE_RE.findall(text_field):
                _add_hs(match)

    seen: set[str] = set()
    extracted_keys: list[str] = []
    for li in items:
        if li.description:
            short = " ".join(li.description.split()[:4])
            if short and short not in seen:
                seen.add(short)
                extracted_keys.append(short)
    return hs_codes, extracted_keys


# ── Granular chunk record builder ─────────────────────────────────────────────

def _build_layout_chunk_records(
    logical_docs: list[LogicalDocument],
    transaction_id: str,
    file_id: str,
    pg_doc_ids: dict[str, str],     # doc_type → postgres UUID (first occurrence wins)
    pg_doc_id_by_doc: list[tuple[LogicalDocument, str]],  # (doc, postgres_uuid)
) -> list[dict]:
    """Build header / table(s) / footer chunk records for every logical document.

    Table chunks are split into groups of MAX_TABLE_ITEMS_PER_CHUNK items.
    Returns records WITHOUT dense/sparse vectors (added by _embed_and_filter).
    """
    max_items = settings.MAX_TABLE_ITEMS_PER_CHUNK
    records: list[dict] = []

    for doc, pg_doc_id in pg_doc_id_by_doc:
        page_nums_json = json.dumps(doc.page_numbers)
        chunk_idx = 0

        # ── Header chunk ─────────────────────────────────────────────────
        header_fields = {k: v for k, v in doc.header_fields.items()
                         if not _is_footer_field(k) and v}
        if header_fields:
            header_text = (
                f"[{doc.doc_type} header]\n"
                + "\n".join(f"{k}: {v}" for k, v in header_fields.items())
            )
            records.append({
                "id":             f"{pg_doc_id}_header_0",
                "transaction_id": transaction_id,
                "file_id":        file_id,
                "document_id":    pg_doc_id,
                "doc_type":       doc.doc_type,
                "page_numbers":   page_nums_json,
                "chunk_type":     "header",
                "chunk_index":    chunk_idx,
                "chunk_text":     header_text[:8192],
                "chunk_metadata": json.dumps({
                    "doc_type":       doc.doc_type,
                    "filename":       doc.filename,
                    "header_fields":  header_fields,
                    "hs_codes":       [],
                    "extracted_keys": list(header_fields.keys()),
                })[:4096],
            })
            chunk_idx += 1

        # ── Table chunks (split into groups) ─────────────────────────────
        if doc.line_items:
            n_items = len(doc.line_items)
            n_chunks = math.ceil(n_items / max_items)

            for batch_num in range(n_chunks):
                start = batch_num * max_items
                end = min(start + max_items, n_items)
                batch = doc.line_items[start:end]

                hs_codes, extracted_keys = _extract_chunk_keys(batch)

                label = (
                    f"[{doc.doc_type} items {start + 1}-{end} of {n_items}]"
                    if n_chunks > 1
                    else f"[{doc.doc_type} items ({n_items} total)]"
                )
                table_text = f"{label}\n{_items_to_markdown_table(batch)}"

                records.append({
                    "id":             f"{pg_doc_id}_table_{chunk_idx}",
                    "transaction_id": transaction_id,
                    "file_id":        file_id,
                    "document_id":    pg_doc_id,
                    "doc_type":       doc.doc_type,
                    "page_numbers":   page_nums_json,
                    "chunk_type":     "table",
                    "chunk_index":    chunk_idx,
                    "chunk_text":     table_text[:8192],
                    "chunk_metadata": json.dumps({
                        "doc_type":        doc.doc_type,
                        "filename":        doc.filename,
                        "hs_codes":        hs_codes,
                        "extracted_keys":  extracted_keys,
                        "table_row_count": len(batch),
                        "item_range":      [start + 1, end],
                    })[:4096],
                })
                chunk_idx += 1

        # ── Footer chunk ─────────────────────────────────────────────────
        footer_fields = {k: v for k, v in doc.header_fields.items()
                         if _is_footer_field(k) and v}
        if footer_fields:
            footer_text = (
                f"[{doc.doc_type} footer]\n"
                + "\n".join(f"{k}: {v}" for k, v in footer_fields.items())
            )
            records.append({
                "id":             f"{pg_doc_id}_footer_{chunk_idx}",
                "transaction_id": transaction_id,
                "file_id":        file_id,
                "document_id":    pg_doc_id,
                "doc_type":       doc.doc_type,
                "page_numbers":   page_nums_json,
                "chunk_type":     "footer",
                "chunk_index":    chunk_idx,
                "chunk_text":     footer_text[:8192],
                "chunk_metadata": json.dumps({
                    "doc_type":       doc.doc_type,
                    "filename":       doc.filename,
                    "footer_fields":  footer_fields,
                    "hs_codes":       [],
                    "extracted_keys": list(footer_fields.keys()),
                })[:4096],
            })

    return records


# ── Embedding helpers (same as pacd_structuring_node) ─────────────────────────

_MIN_EMBED_LEN = 20


def _sanitize_embed_text(text: str) -> str:
    text = text.strip()
    if len(text) < _MIN_EMBED_LEN:
        text = text.ljust(_MIN_EMBED_LEN)
    return text


def _is_valid_dense(vec: list[float]) -> bool:
    return all(math.isfinite(v) for v in vec)


def _embed_and_filter(records: list[dict], text_key: str) -> list[dict]:
    if not records:
        return []
    texts = [_sanitize_embed_text(r[text_key]) for r in records]
    dense_vecs, sparse_vecs = embed_text_chunks(texts)
    valid: list[dict] = []
    for i, record in enumerate(records):
        dv = dense_vecs[i]
        if not _is_valid_dense(dv):
            logger.warning(
                "pacd_layout_chunking_node: dropping record '%s' — NaN/Inf in dense vector",
                record.get("id"),
            )
            continue
        record["dense"] = dv
        record["sparse"] = sparse_vecs[i]
        valid.append(record)
    return valid


# ── LangGraph node ────────────────────────────────────────────────────────────

def pacd_layout_chunking_node(state: GraphState) -> GraphState:
    """LangGraph node (Map phase): chunk PACD pages and index in Milvus.

    Reads
    -----
    state["structured_pages"]   — per-page PageStructuredExtraction dicts
    state["document_id"]        — SHA-256 file identifier (used as file_id)
    state["user_id"], state["transaction_id"], state["filename"]

    Writes
    ------
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("pacd_layout_chunking_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    structured_pages: list[dict] = state.get("structured_pages") or []
    file_id        = state.get("document_id") or ""
    user_id        = state.get("user_id") or ""
    transaction_id = state.get("transaction_id") or ""
    filename       = state.get("filename") or "unknown"

    if not structured_pages:
        errors.append("pacd_layout_chunking_node: no structured_pages in state")
        return {**state, "errors": errors, "current_step": "pacd_layout_chunking_node"}  # type: ignore[return-value]

    # ── Step 1: Merge pages into logical documents ────────────────────────
    logical_docs = _merge_pages_into_documents(structured_pages, filename)
    logger.info(
        "pacd_layout_chunking_node: %d pages → %d logical documents",
        len(structured_pages), len(logical_docs),
    )

    # ── Step 2: Persist to PostgreSQL + collect postgres doc IDs ─────────
    pg_doc_id_by_doc: list[tuple[LogicalDocument, str]] = []
    total_items_created = 0

    for doc in logical_docs:
        try:
            pg_doc_id = postgres_service.insert_pacd_document(
                transaction_id=transaction_id,
                user_id=user_id,
                doc_type=doc.doc_type,
                filename=doc.filename,
                page_numbers=doc.page_numbers,
                header_fields=doc.header_fields,
                raw_ocr_text=doc.raw_ocr_text,
            )
        except Exception as exc:
            errors.append(
                f"pacd_layout_chunking_node: postgres insert failed ({doc.doc_type}): {exc}"
            )
            continue

        if doc.line_items:
            items_data = [
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
                    "raw_text":       li.raw_text,
                }
                for li in doc.line_items
            ]
            try:
                item_ids = postgres_service.insert_pacd_line_items(
                    pacd_document_id=pg_doc_id,
                    transaction_id=transaction_id,
                    items=items_data,
                )
                total_items_created += len(item_ids)
            except Exception as exc:
                errors.append(
                    f"pacd_layout_chunking_node: line item insert failed ({doc.doc_type}): {exc}"
                )

        pg_doc_id_by_doc.append((doc, pg_doc_id))

    logger.info(
        "pacd_layout_chunking_node: persisted %d logical docs, %d line items to Postgres",
        len(pg_doc_id_by_doc), total_items_created,
    )

    # ── Step 3: Build layout-aware chunks and index in Milvus ────────────
    chunk_records = _build_layout_chunk_records(
        logical_docs=logical_docs,
        transaction_id=transaction_id,
        file_id=file_id,
        pg_doc_ids={},
        pg_doc_id_by_doc=pg_doc_id_by_doc,
    )
    logger.info(
        "pacd_layout_chunking_node: built %d chunk records (%d docs × header/tables/footer)",
        len(chunk_records), len(pg_doc_id_by_doc),
    )

    if chunk_records:
        try:
            valid_records = _embed_and_filter(chunk_records, "chunk_text")
            if valid_records:
                pacd_milvus_service.upsert_document_chunks(valid_records)
                logger.info(
                    "pacd_layout_chunking_node: indexed %d/%d chunks in '%s'",
                    len(valid_records), len(chunk_records),
                    pacd_milvus_service.DOCUMENT_CHUNKS_COLLECTION,
                )
        except Exception as exc:
            errors.append(
                f"pacd_layout_chunking_node: Milvus indexing failed — {exc}"
            )
            logger.error("pacd_layout_chunking_node: Milvus error: %s", exc)

    return {**state, "errors": errors, "current_step": "pacd_layout_chunking_node"}  # type: ignore[return-value]
