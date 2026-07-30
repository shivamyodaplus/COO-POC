from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.services import postgres_service
from app.services.document_service import retrieve_document, upload_document

router = APIRouter()
UPLOAD_FILE = File(...)
UPLOAD_COUNTRY = Form(...)
UPLOAD_DOC_TYPE = Form(...)
RETRIEVE_COUNTRY = Form(None)
RETRIEVE_DOC_TYPE = Form(None)
RETRIEVE_TOP_K = Form(3)


@router.post("/upload")
async def upload(
    file: UploadFile = UPLOAD_FILE,
    country: str = UPLOAD_COUNTRY,
    doc_type: str = UPLOAD_DOC_TYPE,
):
    contents = await file.read()
    try:
        result = await asyncio.to_thread(
            upload_document,
            contents,
            file.filename or "upload",
            country.strip(),
            doc_type.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


@router.post("/retrieve")
async def retrieve(
    file: UploadFile = UPLOAD_FILE,
    country: str | None = RETRIEVE_COUNTRY,
    doc_type: str | None = RETRIEVE_DOC_TYPE,
    top_k: int = RETRIEVE_TOP_K,
):
    contents = await file.read()
    try:
        results = await asyncio.to_thread(
            retrieve_document,
            contents,
            file.filename or "query",
            country.strip() if country else None,
            doc_type.strip() if doc_type else None,
            max(1, min(top_k, 20)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return results


@router.get("/images/{image_id}")
async def get_image(image_id: str):
    if "/" in image_id or ".." in image_id:
        raise HTTPException(status_code=400, detail="Invalid image id")

    image = await asyncio.to_thread(postgres_service.get_adapter().get_document_image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    return Response(content=image.data, media_type=image.content_type)
