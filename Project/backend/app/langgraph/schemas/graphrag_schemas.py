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

# Content type ordering for retrieval sort
CONTENT_TYPE_ORDER = {"header": 0, "content": 1, "footer": 2}

# Maximum characters per chunk before splitting
MAX_CHUNK_SIZE = 1000


# ── Phase 1: Page-section chunking schemas ────────────────────────────────────


class PageSections(BaseModel):
    """Three logical sections extracted from a single document page by the VLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    header: str = Field(
        description=(
            "Document-level identifiers at the top of the page: exporter, consignee, "
            "certificate number, dates, invoice numbers, ports, payment terms, etc. "
            "Copy ALL text VERBATIM. Return empty string if no header fields visible."
        ),
    )
    content: str = Field(
        description=(
            "The body/transactional data: line items, goods descriptions, quantities, "
            "weights, HS codes, prices, packing details, totals. "
            "Copy ALL text VERBATIM. Return empty string if no content visible."
        ),
    )
    footer: str = Field(
        description=(
            "Stamps, signatures, issuing authority seals, authentication marks, "
            "certification notes at the bottom of the page. "
            "Copy ALL text VERBATIM. Return empty string if no footer visible."
        ),
    )


# ── Phase 2: Query generation schema ─────────────────────────────────────────


class ValidationQueryList(BaseModel):
    """Up to all targeted verification queries generated from the COO document."""

    queries: list[str] = Field(
        description="Each query asks about one specific field visible on the COO.",
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
    source_doc: str = Field(
        default="",
        description=(
            "Source document name the reference_value was drawn from "
            "(e.g. 'Invoice INVEG25-71198'). Empty string if unknown."
        ),
    )
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
