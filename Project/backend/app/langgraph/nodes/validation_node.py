"""
Validation Node — Phase 2c of 2-Tier GraphRAG.

Takes the transcribed COO text and retrieved reference chunks, then uses the
Text LLM to perform per-attribute verification (PASS/FAIL/UNVERIFIABLE).
Produces a ValidationReport matching the POC's multi-item validation protocol.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.graphrag_schemas import (
    ValidationReport,
    VerificationStatus,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)


def _format_reference_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as structured context for the LLM.

    Chunks are pre-sorted by source_document → page_number → content_type.
    Groups chunks by file and page with visual separators for readability.
    """
    lines = []
    current_source = None
    current_page = None

    for i, c in enumerate(chunks, 1):
        meta = {}
        if c.get("metadata_json"):
            try:
                meta = json.loads(c["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        page = meta.get("page_number", "?")
        content_type = meta.get("content_type", "unknown")
        source = meta.get("source_document", "N/A")

        # Add document separator when transitioning to a new source file
        if source != current_source:
            if current_source is not None:
                lines.append("\n" + "═" * 60)
            lines.append(f"\n══════ DOCUMENT: {source} ══════")
            current_source = source
            current_page = None  # reset page tracking for new doc

        # Add page separator when transitioning to a new page
        if page != current_page:
            if current_page is not None:
                lines.append("\n" + "─" * 60)
            lines.append(f"\n─── Page {page} ───")
            current_page = page

        lines.append(
            f"\n[REF-{i:02d}] [{content_type.upper()}]\n"
            f"{c.get('raw_text', '')}"
        )

    return "\n".join(lines)


def validation_node(state: GraphState) -> dict[str, Any]:
    """Validate COO against retrieved reference chunks."""
    coo_text: str = state.get("coo_text") or ""
    retrieved_chunks: list[dict[str, Any]] = state.get("retrieved_chunks") or []
    transaction_id: str = state.get("transaction_id") or ""

    # Edge case: no chunks retrieved → UNVERIFIABLE
    if not retrieved_chunks:
        report = ValidationReport(
            transaction_id=transaction_id,
            overall_verdict=VerificationStatus.UNVERIFIABLE,
            validated_items=[],
            discrepancies=["No reference chunks retrieved above threshold — cannot validate."],
            summary="No matching reference data found for this transaction.",
        )
        return {
            "verification_report": report.model_dump(),
            "errors": state.get("errors", []),
            "current_step": "validation_node",
        }

    if not coo_text:
        report = ValidationReport(
            transaction_id=transaction_id,
            overall_verdict=VerificationStatus.UNVERIFIABLE,
            validated_items=[],
            discrepancies=["COO text could not be transcribed."],
            summary="COO transcription failed — no text available for validation.",
        )
        return {
            "verification_report": report.model_dump(),
            "errors": state.get("errors", []),
            "current_step": "validation_node",
        }

    # Build the validation prompt
    ref_text = _format_reference_chunks(retrieved_chunks)

    prompt_text = f"""You are a strict trade compliance validator.

══════════════════════════════════════════════════════
MULTI-ITEM VALIDATION PROTOCOL
══════════════════════════════════════════════════════
The COO and reference documents may contain MULTIPLE line items.
The reference data below is organized by page and section (HEADER / CONTENT / FOOTER),
presented in natural reading order for full context.
You MUST follow this protocol to prevent cross-item contamination:

  STEP 1 — ITEM MATCHING:
    Identify each distinct line item in the COO (by description, item number, or HS code).
    Find its MATCHING item in the reference data (match on goods description or HS code).
    If an item in the COO has no matching item in reference → all its fields are UNVERIFIABLE.

  STEP 2 — FIELD VALIDATION (per matched item pair only):
    For each matched COO item ↔ reference item pair, validate EACH field:
      - attribute:       "<field_name> [<item_description>]"
                         e.g. "HS Code [Cocoa Beans]", "Net Weight [Coffee Beans]"
      - coo_value:       value exactly as stated on the COO for THAT item
      - reference_value: value from the MATCHING reference item only
      - source_doc:      the source document name from the page header
                         e.g. "Invoice INVEG25-71198" or "Packing List PL-001"
                         Leave empty string if the source is unknown.
      - status:          PASS / FAIL / UNVERIFIABLE
      - confidence:      0.0–1.0
      - reasoning:       one sentence

  STEP 3 — DOCUMENT-LEVEL FIELDS:
    Validate document-level fields (exporter, consignee, ports, etc.) once,
    not per item.  attribute format: "<field_name> [Document]"
    Fill source_doc the same way as STEP 2.

══════════════════════════════════════════════════════
COMPARISON RULES
══════════════════════════════════════════════════════
  • Case/whitespace: "COCOA BEANS" == "Cocoa Beans" → PASS
  • Partial description: COO "Cocoa Beans, Raw" vs ref "COCOA BEANS, RAW, WHOLE, GRADE I" → PASS
  • Currency prefix: "55,000.00" vs "USD 55,000.00" (same number) → PASS
  • Different quantity units (kg vs Sacs): these are different measurement dimensions
    — mark UNVERIFIABLE with reasoning explaining the unit dimension mismatch
  • Numeric conflict after normalization → FAIL
  • Field-label equivalence: trade documents use different labels for semantically equivalent
    fields (e.g. one document calls a measurement "gross weight" while another calls it
    "net weight"). When a field is absent on the COO under one label but a semantically
    equivalent field IS present under a different label, use that equivalent value as the
    coo_value and note the label difference in reasoning. Only set coo_value to
    "Not specified" when NO weight/quantity measure of any kind is present on the COO
    for that item. Never generate a separate attribute entry for a field that does not
    appear on the COO document at all — skip it rather than marking it UNVERIFIABLE.

overall_verdict: PASS only if all matched fields pass. FAIL if any field fails.
transaction_id MUST be: "{transaction_id}"

COO DOCUMENT TEXT:
{coo_text}

REFERENCE DATA (sorted by page, section: HEADER → CONTENT → FOOTER):
{ref_text}"""

    llm = get_structured_llm(ValidationReport, temperature=0.0)

    try:
        report: ValidationReport = llm.invoke([HumanMessage(content=prompt_text)])  # type: ignore[assignment]
    except Exception as exc:
        logger.error("validation_node: LLM validation failed — %s", exc)
        report = ValidationReport(
            transaction_id=transaction_id,
            overall_verdict=VerificationStatus.UNVERIFIABLE,
            validated_items=[],
            discrepancies=[f"LLM validation call failed: {exc}"],
            summary="Validation could not complete due to an LLM API error.",
        )

    logger.info(
        "validation_node: verdict=%s attributes=%d discrepancies=%d",
        report.overall_verdict.value,
        len(report.validated_items),
        len(report.discrepancies),
    )

    return {
        "verification_report": report.model_dump(),
        "errors": state.get("errors", []),
        "current_step": "validation_node",
    }
