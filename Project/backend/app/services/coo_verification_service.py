"""
COO Verification Service — orchestrates the full COO verification pipeline.

Responsibilities
----------------
1. Accept a raw COO document upload (bytes + filename).
2. Run the COOVerificationWorkflow via LangGraphAdapter.
3. Persist the resulting VerificationReport via transaction_service.
4. Return the final report dict.

Also provides ``run_pacd_ingestion`` for PACD document uploads and
``run_template_attribute_extraction`` for template uploads.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.langgraph.adapters.langgraph_adapter import LangGraphAdapter
from app.langgraph.workflows.coo_verification_workflow import build_coo_verification_workflow
from app.langgraph.workflows.pacd_ingestion_workflow import build_pacd_ingestion_workflow
from app.langgraph.workflows.template_attribute_workflow import build_template_attribute_workflow
from app.services import transaction_service
from app.services.postgres_service import get_adapter

logger = logging.getLogger(__name__)

# Lazy-initialised compiled graphs (one per process)
_coo_workflow: LangGraphAdapter | None = None
_pacd_workflow: LangGraphAdapter | None = None
_template_attr_workflow: LangGraphAdapter | None = None


def _get_coo_workflow() -> LangGraphAdapter:
    global _coo_workflow
    if _coo_workflow is None:
        _coo_workflow = LangGraphAdapter(build_coo_verification_workflow(), "coo_verification_workflow")
    return _coo_workflow


def _get_pacd_workflow() -> LangGraphAdapter:
    global _pacd_workflow
    if _pacd_workflow is None:
        _pacd_workflow = LangGraphAdapter(build_pacd_ingestion_workflow(), "pacd_ingestion_workflow")
    return _pacd_workflow


def _get_template_attr_workflow() -> LangGraphAdapter:
    global _template_attr_workflow
    if _template_attr_workflow is None:
        _template_attr_workflow = LangGraphAdapter(
            build_template_attribute_workflow(), "template_attribute_workflow"
        )
    return _template_attr_workflow


# --------------------------------------------------------------------------- #
# PACD ingestion                                                               #
# --------------------------------------------------------------------------- #

async def run_pacd_ingestion(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    transaction_id: str,
) -> dict[str, Any]:
    """Ingest a PACD document — extract KV pairs and store in Milvus."""
    workflow = _get_pacd_workflow()
    result = await workflow.invoke(
        _initial_state(
            raw_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            transaction_id=transaction_id,
            doc_category="pacd",
        )
    )
    return {
        "document_id": result.get("document_id"),
        "pages_indexed": len(result.get("page_images") or []),
        "extracted_kv_pairs": result.get("extracted_kv_pairs") or [],
        "errors": result.get("errors") or [],
    }


# --------------------------------------------------------------------------- #
# COO verification                                                             #
# --------------------------------------------------------------------------- #

async def run_coo_verification(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    transaction_id: str,
    country: str | None = None,
    doc_type: str | None = None,
) -> dict[str, Any]:
    """Verify a COO document against the transaction's PACD knowledge base."""
    workflow = _get_coo_workflow()
    result = await workflow.invoke(
        _initial_state(
            raw_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            transaction_id=transaction_id,
            doc_category="coo",
            country=country,
            doc_type=doc_type,
        )
    )

    report_data: dict[str, Any] = result.get("verification_report") or {}
    coo_doc_id = result.get("document_id") or ""

    # Persist the report
    if report_data and transaction_id:
        try:
            report_id = transaction_service.save_report(transaction_id, coo_doc_id, report_data)
            report_data["report_id"] = report_id
        except Exception as exc:
            logger.warning("Could not persist verification report: %s", exc)

    return {
        "document_id": coo_doc_id,
        "report": report_data,
        "errors": result.get("errors") or [],
    }


# --------------------------------------------------------------------------- #
# Template attribute extraction                                                #
# --------------------------------------------------------------------------- #

async def run_template_attribute_extraction(
    template_id: str,
    image_bytes: bytes,
    filename: str,
) -> dict[str, str]:
    """Extract field attributes from a template image and persist them."""
    workflow = _get_template_attr_workflow()
    result = await workflow.invoke(
        _initial_state(
            raw_bytes=image_bytes,
            filename=filename,
            doc_category="template",
        )
    )

    attributes: dict[str, str] = result.get("template_attributes") or {}

    if attributes:
        try:
            adapter = get_adapter()
            await asyncio.to_thread(
                adapter.update_visual_template_attributes,
                template_id,
                attributes,
            )
            logger.info("Persisted %d attributes for template %s", len(attributes), template_id)
        except Exception as exc:
            logger.warning("Could not persist template attributes: %s", exc)
            await asyncio.to_thread(adapter.update_template_status, template_id, "failed")
            return attributes

    try:
        adapter = get_adapter()
        await asyncio.to_thread(
            adapter.update_template_status, template_id, "ready" if attributes else "failed"
        )
    except Exception as exc:
        logger.warning("Could not update template status: %s", exc)

    return attributes


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _initial_state(
    *,
    raw_bytes: bytes,
    filename: str,
    user_id: str = "",
    transaction_id: str = "",
    doc_category: str = "pacd",
    country: str | None = None,
    doc_type: str | None = None,
) -> dict[str, Any]:
    """Build a clean initial GraphState for workflow invocation."""
    return {
        "messages": [],
        "raw_bytes": raw_bytes,
        "filename": filename,
        "user_id": user_id,
        "transaction_id": transaction_id,
        "doc_category": doc_category,
        "country": country,
        "doc_type": doc_type,
        "document_id": None,
        "mime_type": None,
        "page_images": [],
        "extracted_fields": {},
        "extracted_kv_pairs": [],
        "retrieved_templates": [],
        "template_attributes": {},
        "confirmed_template_id": None,
        "confirmed_template_name": None,
        "coo_extracted_fields": {},
        "cross_reference_results": [],
        "verification_report": {},
        "errors": [],
        "current_step": None,
    }
