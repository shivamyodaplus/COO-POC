"""
IngestionNode — first node in every document-processing workflow.

Responsibilities
----------------
1. Validate the incoming file (size, MIME type, extension).
2. Assign a deterministic ``document_id``.
3. Load all document pages and convert them to JPEG bytes (``page_images``).
4. Detect MIME type and record page count.
5. Populate the shared ``GraphState`` with ingestion results.

The node reads from and writes to ``GraphState`` only — it never calls the LLM.
"""

from __future__ import annotations

import hashlib
import io
import logging
import tempfile
from pathlib import Path

from app.langgraph.schemas.node_schemas import IngestionNodeInput, IngestionNodeOutput
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

# Mapping of supported extensions → MIME types
_MIME_MAP: dict[str, str] = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def _detect_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return _MIME_MAP.get(ext, "application/octet-stream")


def _make_document_id(filename: str, raw_bytes: bytes) -> str:
    """Deterministic ID: SHA-256 of content + filename stem."""
    digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
    stem = Path(filename).stem.lower().replace(" ", "_")
    return f"{stem}_{digest}"


def _load_page_images(raw_bytes: bytes, filename: str) -> list[bytes]:
    """Load a document and return a list of per-page JPEG bytes."""
    from app.utils.image_processing import load_document_pages

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        pages = load_document_pages(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    jpeg_pages: list[bytes] = []
    for page in pages:
        buf = io.BytesIO()
        page.convert("RGB").save(buf, format="JPEG", quality=85)
        jpeg_pages.append(buf.getvalue())

    return jpeg_pages


def ingestion_node(state: GraphState) -> GraphState:
    """LangGraph node: validate & register the uploaded document.

    Reads
    -----
    state["raw_bytes"], state["filename"]

    Writes
    ------
    state["document_id"], state["mime_type"], state["page_images"],
    state["current_step"]
    Appends to state["errors"] on non-fatal problems.
    """
    logger.info("ingestion_node: starting")

    raw_bytes: bytes | None = state.get("raw_bytes")
    filename: str | None = state.get("filename")

    errors: list[str] = list(state.get("errors") or [])

    # --- Validate input via Pydantic ----------------------------------------
    try:
        node_input = IngestionNodeInput(
            raw_bytes=raw_bytes or b"",
            filename=filename or "",
            country=state.get("country"),
            doc_type=state.get("doc_type"),
        )
    except Exception as exc:
        errors.append(f"ingestion_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "ingestion_node"}  # type: ignore[return-value]

    # --- Core logic ---------------------------------------------------------
    mime_type = _detect_mime(node_input.filename)

    if mime_type == "application/octet-stream":
        errors.append(
            f"ingestion_node: unsupported file extension for '{node_input.filename}'"
        )
        return {**state, "mime_type": mime_type, "errors": errors, "current_step": "ingestion_node"}  # type: ignore[return-value]

    document_id = _make_document_id(node_input.filename, node_input.raw_bytes)

    # Load all pages as JPEG bytes so downstream nodes can use them directly
    page_images: list[bytes] = []
    try:
        page_images = _load_page_images(node_input.raw_bytes, node_input.filename)
    except Exception as exc:
        errors.append(f"ingestion_node: failed to load pages — {exc}")

    # --- Validate output via Pydantic ----------------------------------------
    node_output = IngestionNodeOutput(
        document_id=document_id,
        mime_type=mime_type,
        page_count=len(page_images),
        errors=errors,
    )

    logger.info(
        "ingestion_node: document_id=%s pages=%d",
        node_output.document_id, node_output.page_count,
    )

    return {  # type: ignore[return-value]
        **state,
        "document_id": node_output.document_id,
        "mime_type": node_output.mime_type,
        "page_images": page_images,
        "errors": node_output.errors,
        "current_step": "ingestion_node",
    }
