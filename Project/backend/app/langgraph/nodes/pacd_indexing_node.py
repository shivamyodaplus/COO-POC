"""
PACDIndexingNode — stores PACD document pages into the ``pacd_coo_documents``
Milvus collection using BGE-M3 text embeddings and a KV-pair chunking strategy.

Chunking strategy (per page)
-----------------------------
1. **Individual KV chunks** — each extracted KV pair becomes its own chunk:
   ``"{key}: {value}"``
2. **Grouped KV chunks** — sliding window of CHUNK_WINDOW pairs, stride CHUNK_STRIDE,
   providing contextual overlap.  Duplicated single-pair groups are skipped.

Each chunk is embedded with BGE-M3 (dense 1024-dim + sparse lexical weights)
and stored as one record.  Record IDs use the format::

    {doc_id}_p{page_num}_c{chunk_idx}
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.langgraph.schemas.coo_pacd_schemas import PACDIndexingInput, PACDIndexingOutput
from app.langgraph.state import GraphState
from app.models.embedding import embed_text_chunks
from app.services import pacd_milvus_service, postgres_service
from app.utils.image_processing import preprocess_image

logger = logging.getLogger(__name__)


def _chunk_record_id(document_id: str, page_num: int, chunk_idx: int) -> str:
    return f"{document_id}_p{page_num}_c{chunk_idx}"


def _build_chunks(kv_pairs: list[dict[str, str]], window: int, stride: int) -> list[str]:
    """Build individual + grouped text chunks from a list of KV dicts.

    Returns a list of text strings (one per chunk), deduplicating chunks that
    would be identical to an individual-pair chunk.
    """
    chunks: list[str] = []

    # 1. Individual KV chunks
    for kv in kv_pairs:
        key = str(kv.get("key", "")).strip()
        val = str(kv.get("value", "")).strip()
        if key:
            chunks.append(f"{key}: {val}")

    n = len(kv_pairs)
    if n <= 1:
        return chunks

    # 2. Grouped KV chunks (sliding window)
    for start in range(0, n, stride):
        end = min(start + window, n)
        group = kv_pairs[start:end]
        if len(group) <= 1:
            # Would duplicate an individual chunk — skip
            continue
        group_text = "\n".join(
            f"{kv.get('key', '').strip()}: {kv.get('value', '').strip()}"
            for kv in group
        )
        chunks.append(group_text)
        if end == n:
            break

    return chunks


def pacd_indexing_node(state: GraphState) -> GraphState:
    """LangGraph node: chunk and index PACD document pages into Milvus.

    Reads
    -----
    state["document_id"], state["user_id"], state["transaction_id"],
    state["filename"], state["page_images"], state["extracted_kv_pairs"]

    Writes
    ------
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("pacd_indexing_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    document_id = state.get("document_id") or ""
    user_id = state.get("user_id") or ""
    transaction_id = state.get("transaction_id") or ""
    filename = state.get("filename") or "unknown"
    page_images: list[bytes] = state.get("page_images") or []
    extracted_kv_pairs: list[dict] = state.get("extracted_kv_pairs") or []

    # --- Validate ----------------------------------------------------------------
    try:
        PACDIndexingInput(
            document_id=document_id,
            user_id=user_id,
            transaction_id=transaction_id,
            filename=filename,
            extracted_kv_pairs=extracted_kv_pairs,
            page_images=page_images,
        )
    except Exception as exc:
        errors.append(f"pacd_indexing_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "pacd_indexing_node"}  # type: ignore[return-value]

    # --- Group kv pairs by page ------------------------------------------------
    kv_by_page: dict[int, list[dict[str, str]]] = {}
    for kv in extracted_kv_pairs:
        page = int(kv.get("page", 1))
        kv_by_page.setdefault(page, []).append({"key": kv["key"], "value": kv["value"]})

    # --- Build chunks and embed per page ----------------------------------------
    all_milvus_records: list[dict[str, Any]] = []
    postgres_records: list[dict[str, Any]] = []
    window = settings.CHUNK_WINDOW
    stride = settings.CHUNK_STRIDE

    for page_num, image_bytes in enumerate(page_images, start=1):
        page_kv = kv_by_page.get(page_num, [])
        if not page_kv:
            logger.debug("pacd_indexing_node: no KV pairs for page %d — skipping", page_num)
            continue

        chunks = _build_chunks(page_kv, window, stride)
        if not chunks:
            continue

        # Embed all chunks for this page in one batch call
        try:
            dense_vecs, sparse_vecs = embed_text_chunks(chunks)
        except Exception as exc:
            errors.append(f"pacd_indexing_node p{page_num}: BGE-M3 embedding failed — {exc}")
            continue

        for chunk_idx, (chunk_text, dense, sparse) in enumerate(
            zip(chunks, dense_vecs, sparse_vecs)
        ):
            record_id = _chunk_record_id(document_id, page_num, chunk_idx)
            all_milvus_records.append({
                "id":             record_id,
                "user_id":        user_id,
                "transaction_id": transaction_id,
                "doc_category":   "pacd",
                "filename":       filename,
                "page_num":       page_num,
                "chunk_index":    chunk_idx,
                "chunk_text":     chunk_text[:4096],
                "dense":          dense,
                "sparse":         sparse,
            })

        # Store first-page image in Postgres for audit / retrieval
        try:
            from PIL import Image
            import io
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            preprocessed = preprocess_image(pil_image)
            import io as _io
            buf = _io.BytesIO()
            preprocessed.save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()
        except Exception:
            jpeg_bytes = image_bytes

        postgres_records.append({
            "id":                 f"{document_id}_p{page_num}",
            "country":            "",
            "doc_type":           "pacd",
            "file_name":          filename,
            "ocr_text":           " ".join(
                f"{kv['key']}: {kv['value']}" for kv in page_kv
            )[:65535],
            "image_data":         jpeg_bytes,
            "image_content_type": "image/jpeg",
            # Note: no 'dense' here — PACD Milvus indexing is handled
            # exclusively by pacd_milvus_service via all_milvus_records above.
        })

    # --- Persist ---------------------------------------------------------------
    if all_milvus_records:
        try:
            pacd_milvus_service.insert_pacd_chunks(all_milvus_records)
            logger.info(
                "pacd_indexing_node: inserted %d chunks into Milvus (tx=%s)",
                len(all_milvus_records), transaction_id,
            )
        except Exception as exc:
            errors.append(f"pacd_indexing_node: Milvus insert failed — {exc}")
    else:
        logger.warning(
            "pacd_indexing_node: 0 chunks to index for tx=%s — "
            "vision_extraction returned %d KV pairs across %d pages",
            transaction_id, len(extracted_kv_pairs), len(page_images),
        )

    if postgres_records:
        try:
            postgres_service.insert_documents_with_outbox(postgres_records)
            logger.info(
                "pacd_indexing_node: inserted %d pages into Postgres", len(postgres_records)
            )
        except Exception as exc:
            errors.append(f"pacd_indexing_node: Postgres insert failed — {exc}")

    output = PACDIndexingOutput(
        document_id=document_id,
        pages_indexed=len({r["page_num"] for r in all_milvus_records}),
        errors=errors,
    )

    return {  # type: ignore[return-value]
        **state,
        "errors": output.errors,
        "pages_indexed": output.pages_indexed,
        "current_step": "pacd_indexing_node",
    }
