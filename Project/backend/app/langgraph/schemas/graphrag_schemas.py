"""
GraphRAG Schemas — 2-Tier Decoupled GraphRAG for COO Validation.

Phase 1 (PACD Ingest): VLM logical chunking → BGE-M3 embed → Milvus
Phase 2 (COO Validate): VLM transcription → query generation → hybrid search → validation
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# ── VARCHAR field length limits (must match Milvus schema) ────────────────────
MAX_CHUNK_TOPIC = 200


# ── Phase 1: Document chunking schemas ────────────────────────────────────────


class DocumentChunk(BaseModel):
    """One logical section of a trade document as identified by the VLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    chunk_topic: str = Field(
        max_length=MAX_CHUNK_TOPIC,
        description=(
            f"Short label (≤{MAX_CHUNK_TOPIC} chars) describing what this section is about. "
            "Examples: 'Invoice Header', 'Exporter Details', 'Line Item — Cocoa Beans', "
            "'Invoice Totals', 'Packing Details'."
        ),
    )
    raw_text: str = Field(
        description="Verbatim text content of this chunk. Include ALL text — never truncate or omit data.",
    )


class DocumentChunkList(BaseModel):
    """All logical chunks extracted from a single document (or set of pages)."""

    chunks: list[DocumentChunk]


# ── Phase 2: Query generation schema ─────────────────────────────────────────


class ValidationQueryList(BaseModel):
    """Up to 10 targeted verification queries generated from the COO document."""

    queries: list[str] = Field(
        description="Each query asks about one specific field visible on the COO.",
        max_length=10,
    )


# ── Phase 2: Validation result schemas ───────────────────────────────────────


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIABLE = "UNVERIFIABLE"


class AttributeResult(BaseModel):
    """Verification result for a single field on the COO."""

    attribute: str = Field(description="Name of the field being validated")
    coo_value: str = Field(description="Value as stated on the COO")
    reference_value: str = Field(description="Value found in the reference documents")
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One sentence justification")


class ValidationReport(BaseModel):
    """Final validation report comparing COO against reference documents."""

    transaction_id: str
    overall_verdict: VerificationStatus
    validated_items: list[AttributeResult]
    discrepancies: list[str] = Field(description="Human-readable list of mismatches")
    summary: str = Field(description="One-paragraph natural language summary")
