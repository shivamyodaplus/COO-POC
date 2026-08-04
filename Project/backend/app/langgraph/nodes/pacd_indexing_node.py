"""
PACDIndexingNode — stores PACD document pages into:
  1. The ``pacd_coo_documents`` Milvus collection (for cross-reference search).
  2. Postgres ``documents`` table (for image storage + outbox sync).

One Milvus record is created per page.  The ``extracted_kv`` field holds a
JSON-serialised list of {key, value} pairs from the Vision extraction step.
Dense embeddings are computed from the page images via the existing
``embed_image`` model.
"""

from __future__ import annotations

import json
import logging

from app.langgraph.schemas.coo_pacd_schemas import PACDIndexingInput, PACDIndexingOutput
from app.langgraph.state import GraphState
from app.models.embedding import embed_image
from app.models.ocr import extract_text
from app.services import pacd_milvus_service, postgres_service
from app.utils.image_processing import preprocess_image

logger = logging.getLogger(__name__)


def _page_record_id(document_id: str, page_num: int) -> str:
    return f"{document_id}_p{page_num}"


def pacd_indexing_node(state: GraphState) -> GraphState:
    """LangGraph node: index PACD document pages into Milvus + Postgres.

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
    kv_by_page: dict[int, list[dict]] = {}
    for kv in extracted_kv_pairs:
        page = kv.get("page", 1)
        kv_by_page.setdefault(page, []).append({"key": kv["key"], "value": kv["value"]})

    # --- Build Milvus + Postgres records per page -----------------------------
    milvus_records: list[dict] = []
    postgres_records: list[dict] = []

    for page_num, image_bytes in enumerate(page_images, start=1):
        try:
            from PIL import Image
            import io
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            preprocessed = preprocess_image(pil_image)
            dense_vec = embed_image(preprocessed).tolist()
            ocr_text = extract_text(preprocessed)
        except Exception as exc:
            errors.append(f"pacd_indexing_node p{page_num}: embedding failed — {exc}")
            continue

        page_kv = kv_by_page.get(page_num, [])
        record_id = _page_record_id(document_id, page_num)

        milvus_records.append({
            "id":             record_id,
            "user_id":        user_id,
            "transaction_id": transaction_id,
            "doc_category":   "pacd",
            "filename":       filename,
            "page_num":       page_num,
            "ocr_text":       ocr_text or "",
            "extracted_kv":   json.dumps(page_kv),
            "dense":          dense_vec,
        })

        postgres_records.append({
            "id":                  record_id,
            "country":             "",
            "doc_type":            "pacd",
            "file_name":           filename,
            "ocr_text":            ocr_text or "",
            "image_data":          image_bytes,
            "image_content_type":  "image/jpeg",
            "dense":               dense_vec,
        })

    # --- Persist ---------------------------------------------------------------
    if milvus_records:
        try:
            pacd_milvus_service.insert_pacd_pages(milvus_records)
            logger.info("pacd_indexing_node: inserted %d pages into Milvus", len(milvus_records))
        except Exception as exc:
            errors.append(f"pacd_indexing_node: Milvus insert failed — {exc}")

    if postgres_records:
        try:
            postgres_service.insert_documents_with_outbox(postgres_records)
            logger.info("pacd_indexing_node: inserted %d pages into Postgres", len(postgres_records))
        except Exception as exc:
            errors.append(f"pacd_indexing_node: Postgres insert failed — {exc}")

    output = PACDIndexingOutput(
        document_id=document_id,
        pages_indexed=len(milvus_records),
        errors=errors,
    )

    return {  # type: ignore[return-value]
        **state,
        "errors": output.errors,
        "current_step": "pacd_indexing_node",
    }
