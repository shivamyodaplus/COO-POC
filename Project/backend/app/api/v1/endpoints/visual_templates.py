from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.services import postgres_service
from app.utils.db_health import assert_databases_ready
from app.utils.image_processing import load_document_pages
from app.utils.template_matching import match_templates

router = APIRouter()
VALID_TEMPLATE_TYPES = {"sign", "signature", "stamp", "logo"}
VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}
UPLOAD_FILE = File(...)
UPLOAD_NAME = Form(...)
UPLOAD_TEMPLATE_TYPE = Form(...)
UPLOAD_COUNTRY = Form(None)
UPLOAD_DOC_TYPE = Form(None)
MATCH_TEMPLATE_IDS  = Form(None)
MATCH_COUNTRY       = Form(None)
MATCH_DOC_TYPE      = Form(None)
MATCH_THRESHOLD     = Form(0.2)
MATCH_P2_THRESHOLD  = Form(10)


def _decode_image_bytes(raw: bytes) -> np.ndarray | None:
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _load_query_image(file_bytes: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        pages = load_document_pages(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not pages:
        raise ValueError("No pages found in uploaded document")

    query_rgb = np.array(pages[0].convert("RGB"))
    return cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)


def _encode_jpeg_base64(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Failed to encode query preview")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@router.post("/upload")
async def upload_visual_template(
    file: UploadFile = UPLOAD_FILE,
    name: str = UPLOAD_NAME,
    template_type: str = UPLOAD_TEMPLATE_TYPE,
    country: str | None = UPLOAD_COUNTRY,
    doc_type: str | None = UPLOAD_DOC_TYPE,
):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in VALID_IMAGE_SUFFIXES:
        raise HTTPException(status_code=422, detail=f"Unsupported visual template type: {suffix}")

    template_type_normalized = template_type.strip().lower()
    if template_type_normalized not in VALID_TEMPLATE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"template_type must be one of: {', '.join(sorted(VALID_TEMPLATE_TYPES))}",
        )

    assert_databases_ready()

    content = await file.read()
    adapter = postgres_service.get_adapter()
    record = await asyncio.to_thread(
        adapter.save_visual_template,
        name.strip() or Path(file.filename or "template").stem,
        template_type_normalized,
        country.strip() if country and country.strip() else None,
        doc_type.strip() if doc_type and doc_type.strip() else None,
        content,
        file.content_type or "application/octet-stream",
    )

    # Fire-and-forget: extract template attributes in the background (non-blocking)
    asyncio.create_task(
        _extract_and_persist_template_attributes(
            template_id=record.id,
            image_bytes=content,
            filename=file.filename or "template",
        )
    )

    return {
        "id": record.id,
        "name": record.name,
        "template_type": record.template_type,
        "country": record.country,
        "doc_type": record.doc_type,
        "created_at": record.created_at,
        "extracted_attributes": None,  # populated async in background
    }


async def _extract_and_persist_template_attributes(
    template_id: str,
    image_bytes: bytes,
    filename: str,
) -> None:
    """Background task: run TemplateAttributeWorkflow and persist results."""
    try:
        from app.services.coo_verification_service import run_template_attribute_extraction
        await run_template_attribute_extraction(
            template_id=template_id,
            image_bytes=image_bytes,
            filename=filename,
        )
    except Exception:
        pass  # non-fatal — template still usable without extracted_attributes


@router.get("")
async def list_visual_templates(
    country: str | None = None,
    doc_type: str | None = None,
):
    adapter = postgres_service.get_adapter()
    items = await asyncio.to_thread(
        adapter.list_visual_templates,
        country.strip() if country and country.strip() else None,
        doc_type.strip() if doc_type and doc_type.strip() else None,
    )
    return [
        {
            "id": item.id,
            "name": item.name,
            "template_type": item.template_type,
            "country": item.country,
            "doc_type": item.doc_type,
            "created_at": item.created_at,
        }
        for item in items
    ]


@router.get("/{template_id}/image")
async def get_visual_template_image(template_id: str):
    adapter = postgres_service.get_adapter()
    image = await asyncio.to_thread(adapter.get_visual_template_image, template_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Visual template not found")
    return Response(content=image.data, media_type=image.content_type)


@router.delete("/{template_id}")
async def delete_visual_template(template_id: str):
    adapter = postgres_service.get_adapter()
    deleted = await asyncio.to_thread(adapter.delete_visual_template, template_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Visual template not found")
    return {"deleted": True, "id": template_id}


@router.post("/match")
async def match_visual_templates(
    file: UploadFile = UPLOAD_FILE,
    template_ids: list[str] | None = MATCH_TEMPLATE_IDS,
    country: str | None = MATCH_COUNTRY,
    doc_type: str | None = MATCH_DOC_TYPE,
    threshold: float = MATCH_THRESHOLD,
    p2_threshold: int = MATCH_P2_THRESHOLD,
):
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")

    adapter = postgres_service.get_adapter()
    file_bytes = await file.read()

    try:
        query_image = await asyncio.to_thread(_load_query_image, file_bytes, file.filename or "query")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    selected_templates: list[dict[str, str | None]] = []
    if template_ids:
        for template_id in template_ids:
            meta = await asyncio.to_thread(adapter.get_visual_template, template_id)
            image = await asyncio.to_thread(adapter.get_visual_template_image, template_id)
            if meta is None or image is None:
                continue
            selected_templates.append(
                {
                    "id": meta.id,
                    "name": meta.name,
                    "template_type": meta.template_type,
                    "country": meta.country,
                    "doc_type": meta.doc_type,
                    "created_at": meta.created_at,
                    "image_bytes": image.data,
                }
            )
    else:
        listed = await asyncio.to_thread(
            adapter.list_visual_templates,
            country.strip() if country and country.strip() else None,
            doc_type.strip() if doc_type and doc_type.strip() else None,
        )
        for item in listed:
            image = await asyncio.to_thread(adapter.get_visual_template_image, item.id)
            if image is None:
                continue
            selected_templates.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "template_type": item.template_type,
                    "country": item.country,
                    "doc_type": item.doc_type,
                    "created_at": item.created_at,
                    "image_bytes": image.data,
                }
            )

    if not selected_templates:
        raise HTTPException(status_code=404, detail="No visual templates available for matching")

    template_images: list[np.ndarray] = []
    filtered_templates: list[dict[str, str | None]] = []
    for item in selected_templates:
        decoded = _decode_image_bytes(item["image_bytes"])
        if decoded is None:
            continue
        filtered_templates.append(item)
        template_images.append(decoded)

    if not template_images:
        raise HTTPException(status_code=422, detail="Unable to decode visual templates")

    output = await asyncio.to_thread(
        match_templates,
        query_image,
        template_images,
        threshold,
        None,  # scales — use default
        30,    # canny_low
        100,   # canny_high
        p2_threshold,
    )

    try:
        query_preview_b64 = await asyncio.to_thread(_encode_jpeg_base64, query_image)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    results: list[dict[str, str | float | bool | list[int] | None]] = []
    for idx, match in enumerate(output.results):
        template = filtered_templates[idx]
        results.append(
            {
                "template_id": template["id"],
                "name": template["name"],
                "template_type": template["template_type"],
                "country": template["country"],
                "doc_type": template["doc_type"],
                "found": match.found,
                "score": match.score,
                "scale": match.scale,
                "bounding_box": list(match.bounding_box),
                "p2_verdict": match.p2_verdict,
                "p2_n_matches": match.p2_n_matches,
                "p2_n_inliers": match.p2_n_inliers,
                "p2_homography_vis_jpeg_b64": match.p2_homography_vis_jpeg_b64,
                "p2_refined_jpeg_b64": match.p2_refined_jpeg_b64,
            }
        )

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    return {
        "any_found": any(item["found"] for item in results),
        "count": len(results),
        "results": results,
        "query_preview_jpeg_base64": query_preview_b64,
        "query_preview_content_type": "image/jpeg",
    }
