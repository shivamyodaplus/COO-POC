from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.services.document_service import IMAGE_DIR, retrieve_document, upload_document

router = APIRouter()


@router.post("/upload")
async def upload(
    file: Annotated[UploadFile, File()],
    country: Annotated[str, Form()],
    doc_type: Annotated[str, Form()],
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
    file: Annotated[UploadFile, File()],
    country: Annotated[str | None, Form()] = None,
    doc_type: Annotated[str | None, Form()] = None,
    top_k: Annotated[int, Form()] = 3,
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
    path = IMAGE_DIR / f"{image_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")
