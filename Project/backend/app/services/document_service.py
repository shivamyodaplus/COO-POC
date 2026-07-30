from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.embedding import embed_image
from app.models.ocr import extract_text
from app.services import milvus_service, postgres_service
from app.utils.db_health import assert_databases_ready
from app.utils.image_processing import (
    SUPPORTED_EXTENSIONS,
    load_document_pages,
    preprocess_image,
)


def _make_id(filename: str, page_num: int, country: str, doc_type: str) -> str:
    stem = Path(filename).stem
    base = f"{country}_{doc_type}_{stem}_p{page_num}"
    return base.lower().replace(" ", "_")


def _to_jpeg_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def upload_document(
    file_bytes: bytes,
    filename: str,
    country: str,
    doc_type: str,
) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        pages = load_document_pages(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    adapter = postgres_service.get_adapter()
    use_postgres = settings.STORAGE_ADAPTER.lower() == "postgres"

    if use_postgres:
        # Guard the transaction boundary by checking both data stores first.
        assert_databases_ready()

    records: list[dict[str, Any]] = []
    tx_records: list[dict[str, Any]] = []
    for page_num, image in enumerate(pages, start=1):
        preprocessed = preprocess_image(image)
        dense_vec = embed_image(preprocessed).tolist()
        ocr_text = extract_text(preprocessed)
        img_id = _make_id(filename, page_num, country, doc_type)
        image_bytes = _to_jpeg_bytes(preprocessed)

        if use_postgres:
            tx_records.append({
                "id": img_id,
                "country": country,
                "doc_type": doc_type,
                "file_name": filename,
                "ocr_text": ocr_text,
                "dense": dense_vec,
                "image_data": image_bytes,
                "image_content_type": "image/jpeg",
            })
        else:
            adapter.save_document_image(img_id, image_bytes, "image/jpeg")
            records.append({
                "id": img_id,
                "country": country,
                "doc_type": doc_type,
                "file_name": filename,
                "ocr_text": ocr_text,
                "dense": dense_vec,
                # sparse is generated server-side by Milvus BM25 function
            })

    if use_postgres:
        postgres_service.insert_documents_with_outbox(tx_records)
    else:
        milvus_service.insert(records)

    return {
        "id": (tx_records[0]["id"] if tx_records else (records[0]["id"] if records else "")),
        "pages_stored": len(tx_records) if tx_records else len(records),
        "country": country,
        "doc_type": doc_type,
        "filename": filename,
    }


def retrieve_document(
    file_bytes: bytes,
    filename: str,
    country: str | None,
    doc_type: str | None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        pages = load_document_pages(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Use first page as the query image
    preprocessed = preprocess_image(pages[0])
    dense_vec = embed_image(preprocessed).tolist()
    ocr_text = extract_text(preprocessed)

    return milvus_service.hybrid_search(
        dense_vec=dense_vec,
        ocr_text=ocr_text,
        country=country,
        doc_type=doc_type,
        top_k=top_k,
    )
