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

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_llm
from app.langgraph.schemas.graphrag_schemas import (
    ValidationReport,
    VerificationStatus,
)
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)


def _format_reference_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as structured context for the LLM."""
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = {}
        if c.get("metadata_json"):
            try:
                meta = json.loads(c["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        lines.append(
            f"[REF-{i:02d}] score={c.get('score', 0):.3f}  "
            f"topic='{meta.get('chunk_topic', 'N/A')}'  "
            f"source='{meta.get('source_document', 'N/A')}'\n"
            f"{c.get('raw_text', '')}"
        )
    return "\n\n".join(lines)


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
    schema_json = json.dumps(ValidationReport.model_json_schema(), indent=2)
    ref_text = _format_reference_chunks(retrieved_chunks)

    prompt_text = f"""You are a strict trade compliance validator.

══════════════════════════════════════════════════════
MULTI-ITEM VALIDATION PROTOCOL
══════════════════════════════════════════════════════
The COO and reference documents may contain MULTIPLE line items.
You MUST follow this protocol to prevent cross-item contamination:

  STEP 1 — ITEM MATCHING:
    Identify each distinct line item in the COO (by description, item number, or HS code).
    Find its MATCHING item in the reference chunks (match on goods description or HS code).
    If an item in the COO has no matching item in reference → all its fields are UNVERIFIABLE.

  STEP 2 — FIELD VALIDATION (per matched item pair only):
    For each matched COO item ↔ reference item pair, validate EACH field:
      - attribute:       "<field_name> [<item_description>]"
                         e.g. "HS Code [Cocoa Beans]", "Net Weight [Coffee Beans]"
      - coo_value:       value exactly as stated on the COO for THAT item
      - reference_value: value from the MATCHING reference item only
      - status:          PASS / FAIL / UNVERIFIABLE
      - confidence:      0.0–1.0
      - reasoning:       one sentence

  STEP 3 — DOCUMENT-LEVEL FIELDS:
    Validate document-level fields (exporter, consignee, ports, etc.) once,
    not per item.  attribute format: "<field_name> [Document]"

══════════════════════════════════════════════════════
COMPARISON RULES
══════════════════════════════════════════════════════
  • Case/whitespace: "COCOA BEANS" == "Cocoa Beans" → PASS
  • Partial description: COO "Cocoa Beans, Raw" vs ref "COCOA BEANS, RAW, WHOLE, GRADE I" → PASS
  • Currency prefix: "55,000.00" vs "USD 55,000.00" (same number) → PASS
  • Different quantity units (kg vs Sacs): these are different measurement dimensions
    — mark UNVERIFIABLE with reasoning explaining the unit dimension mismatch
  • Numeric conflict after normalization → FAIL

overall_verdict: PASS only if all matched fields pass. FAIL if any field fails.
transaction_id MUST be: "{transaction_id}"

Respond ONLY with a JSON object matching this schema:
{schema_json}

COO DOCUMENT TEXT:
{coo_text}

REFERENCE CHUNKS:
{ref_text}"""

    system_text = (
        "You are a precise data extraction assistant. "
        "You MUST respond with valid JSON that matches the schema exactly. "
        "Return ONLY the JSON object, no markdown fences, no extra text."
    )

    llm = get_llm(temperature=0.0)

    try:
        response = llm.invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=prompt_text),
        ])
        raw_json = response.content.strip()

        # Handle markdown fences if present
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw_json)
        report = ValidationReport(**parsed)

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
