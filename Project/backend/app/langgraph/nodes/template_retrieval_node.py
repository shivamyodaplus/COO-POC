"""
TemplateRetrievalNode — retrieves the top-k most visually similar templates
from the existing Milvus ``templates`` collection using the first page of the
COO document as the query image.

This reuses the existing ``milvus_service.hybrid_search`` pipeline.
"""

from __future__ import annotations

import io
import logging

from app.langgraph.schemas.coo_pacd_schemas import (
    TemplateRetrievalInput,
    TemplateRetrievalOutput,
)
from app.langgraph.state import GraphState
from app.models.embedding import embed_image
from app.models.ocr import extract_text
from app.services.milvus_service import hybrid_search
from app.utils.image_processing import preprocess_image
from app.core.config import settings

logger = logging.getLogger(__name__)


def template_retrieval_node(state: GraphState) -> GraphState:
    """LangGraph node: retrieve top-k template candidates for the COO page.

    Reads
    -----
    state["page_images"]  — first JPEG is used as the query
    state["document_id"], state["country"], state["doc_type"]

    Writes
    ------
    state["retrieved_templates"]  — list of candidate template dicts
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("template_retrieval_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    page_images: list[bytes] = state.get("page_images") or []

    if not page_images:
        errors.append("template_retrieval_node: no page_images in state")
        return {**state, "errors": errors, "current_step": "template_retrieval_node"}  # type: ignore[return-value]

    try:
        node_input = TemplateRetrievalInput(
            document_id=state.get("document_id") or "",
            country=state.get("country"),
            doc_type=state.get("doc_type"),
            top_k=settings.COO_TEMPLATE_TOP_K,
        )
    except Exception as exc:
        errors.append(f"template_retrieval_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "template_retrieval_node"}  # type: ignore[return-value]

    # Build query embedding from the first page
    try:
        from PIL import Image
        pil_image = Image.open(io.BytesIO(page_images[0])).convert("RGB")
        preprocessed = preprocess_image(pil_image)
        dense_vec = embed_image(preprocessed).tolist()
        ocr_text = extract_text(preprocessed)
    except Exception as exc:
        errors.append(f"template_retrieval_node: embedding failed — {exc}")
        return {**state, "errors": errors, "current_step": "template_retrieval_node"}  # type: ignore[return-value]

    try:
        candidates = hybrid_search(
            dense_vec=dense_vec,
            ocr_text=ocr_text,
            country=node_input.country,
            doc_type=node_input.doc_type,
            top_k=node_input.top_k,
        )
    except Exception as exc:
        errors.append(f"template_retrieval_node: Milvus search failed — {exc}")
        candidates = []

    output = TemplateRetrievalOutput(candidates=candidates, errors=errors)
    logger.info("template_retrieval_node: retrieved %d candidates", len(output.candidates))

    return {  # type: ignore[return-value]
        **state,
        "retrieved_templates": output.candidates,
        "errors": output.errors,
        "current_step": "template_retrieval_node",
    }
