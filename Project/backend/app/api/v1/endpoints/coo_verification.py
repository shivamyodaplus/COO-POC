"""
COO Verification API — PACD ingestion and COO cross-reference endpoints.

Endpoints
---------
POST /transactions/{tx_id}/pacd  — upload a PACD document into the knowledge base
POST /transactions/{tx_id}/coo   — upload a COO document and run cross-reference
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.coo_verification_service import run_coo_verification, run_pacd_ingestion
from app.services.transaction_service import get_transaction
from app.utils.db_health import assert_databases_ready

router = APIRouter()

_FILE = File(...)
_USER_ID = Form(...)
_COUNTRY = Form(None)
_DOC_TYPE = Form(None)


@router.post("/{transaction_id}/pacd")
async def upload_pacd(
    transaction_id: str,
    file: UploadFile = _FILE,
    user_id: str = _USER_ID,
):
    """Ingest a PACD document — Vision LLM extracts KV pairs and stores in Milvus."""
    assert_databases_ready()

    tx = await asyncio.to_thread(get_transaction, transaction_id)
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    file_bytes = await file.read()
    try:
        result = await run_pacd_ingestion(
            file_bytes=file_bytes,
            filename=file.filename or "pacd_upload",
            user_id=user_id.strip(),
            transaction_id=transaction_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PACD ingestion failed: {exc}")

    return {
        "transaction_id": transaction_id,
        "document_id": result.get("document_id"),
        "pages_indexed": result.get("pages_indexed", 0),
        "extracted_kv_pairs": result.get("extracted_kv_pairs") or [],
        "errors": result.get("errors") or [],
    }


@router.post("/{transaction_id}/coo")
async def verify_coo(
    transaction_id: str,
    file: UploadFile = _FILE,
    user_id: str = _USER_ID,
    country: str | None = _COUNTRY,
    doc_type: str | None = _DOC_TYPE,
):
    """Upload a COO document and run the full cross-reference verification pipeline."""
    assert_databases_ready()

    tx = await asyncio.to_thread(get_transaction, transaction_id)
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    file_bytes = await file.read()
    try:
        result = await run_coo_verification(
            file_bytes=file_bytes,
            filename=file.filename or "coo_upload",
            user_id=user_id.strip(),
            transaction_id=transaction_id,
            country=country.strip() if country else None,
            doc_type=doc_type.strip() if doc_type else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"COO verification failed: {exc}")

    return {
        "transaction_id": transaction_id,
        "document_id": result.get("document_id"),
        "report": result.get("report") or {},
        "errors": result.get("errors") or [],
    }
