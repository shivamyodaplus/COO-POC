"""
Pydantic schemas for the COO / PACD cross-reference verification pipeline.

Every node in the pipeline validates its input and output against one of the
models defined here.  All models inherit ``NodeBase`` (strict, no extra fields).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #

class NodeBase(BaseModel):
    model_config = {
        "strict": False,   # allow coercion (e.g. int → float) for convenience
        "extra": "forbid",
        "populate_by_name": True,
    }


# --------------------------------------------------------------------------- #
# Vision extraction (PACD page / template attribute)                          #
# --------------------------------------------------------------------------- #

class PageKVPair(NodeBase):
    key: str = Field(..., description="Field name extracted from the document image.")
    value: str = Field(..., description="Extracted field value as a string.")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class VisionExtractionInput(NodeBase):
    document_id: str
    page_num: int = Field(..., ge=1)
    image_b64: str = Field(..., description="Base64-encoded JPEG of the page.")
    doc_category: Literal["pacd", "coo", "template"] = "pacd"
    field_hints: list[str] = Field(
        default_factory=list,
        description="Optional list of field names to focus on (from confirmed template).",
    )


class VisionExtractionOutput(NodeBase):
    document_id: str
    page_num: int
    kv_pairs: list[PageKVPair] = Field(default_factory=list)
    raw_llm_response: str | None = None
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# PACD indexing                                                                #
# --------------------------------------------------------------------------- #

class PACDIndexingInput(NodeBase):
    document_id: str
    user_id: str
    transaction_id: str
    filename: str
    extracted_kv_pairs: list[dict[str, Any]]  # [{page, key, value, confidence}]
    page_images: list[bytes]                  # per-page JPEG bytes


class PACDIndexingOutput(NodeBase):
    document_id: str
    pages_indexed: int
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Template attribute extraction                                                #
# --------------------------------------------------------------------------- #

class TemplateAttributeInput(NodeBase):
    template_id: str
    template_name: str
    image_b64: str = Field(..., description="Base64-encoded JPEG of the template image.")


class TemplateAttributeOutput(NodeBase):
    template_id: str
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of attribute_key → description / label as seen on the template.",
    )
    raw_llm_response: str | None = None
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Template retrieval & confirmation                                            #
# --------------------------------------------------------------------------- #

class TemplateRetrievalInput(NodeBase):
    document_id: str
    country: str | None = None
    doc_type: str | None = None
    top_k: int = Field(3, ge=1, le=10)


class TemplateRetrievalOutput(NodeBase):
    candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top-k template records [{id, name, template_type, score, ...}].",
    )
    errors: list[str] = Field(default_factory=list)


class TemplateConfirmationInput(NodeBase):
    document_id: str
    coo_image_b64: str = Field(..., description="Base64-encoded JPEG of the COO page.")
    candidates: list[dict[str, Any]] = Field(
        ..., description="Template candidates returned by retrieval."
    )


class TemplateConfirmationOutput(NodeBase):
    confirmed_template_id: str | None = None
    confirmed_template_name: str | None = None
    reasoning: str | None = None
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# COO extraction                                                               #
# --------------------------------------------------------------------------- #

class COOExtractionInput(NodeBase):
    document_id: str
    coo_image_b64: str = Field(..., description="Base64-encoded JPEG of the COO page.")
    template_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Field schema from the confirmed template (key → label).",
    )


class COOExtractionOutput(NodeBase):
    coo_extracted_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Flat dict of extracted field values from the COO document.",
    )
    raw_llm_response: str | None = None
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Cross-reference                                                              #
# --------------------------------------------------------------------------- #

class DiscrepancyItem(NodeBase):
    field_key: str
    coo_value: str | None = None
    pacd_value: str | None = None
    verdict: Literal["match", "mismatch", "not_found_in_pacd", "not_in_coo"] = "not_found_in_pacd"
    pacd_source_doc: str | None = Field(None, description="Document ID where PACD value was found.")
    pacd_source_page: int | None = None


class CrossReferenceInput(NodeBase):
    transaction_id: str
    user_id: str
    coo_extracted_fields: dict[str, str]
    scope: Literal["single_transaction", "full_history"] = "single_transaction"
    # full_history: include all transactions for this user_id (future use)


class CrossReferenceOutput(NodeBase):
    discrepancy_table: list[DiscrepancyItem] = Field(default_factory=list)
    total_fields: int = 0
    matched: int = 0
    mismatched: int = 0
    not_found: int = 0
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Verification report                                                          #
# --------------------------------------------------------------------------- #

class VerificationReport(NodeBase):
    transaction_id: str
    coo_doc_id: str
    confirmed_template_id: str | None = None
    confirmed_template_name: str | None = None
    discrepancy_table: list[DiscrepancyItem] = Field(default_factory=list)
    narrative: str = ""
    summary: str = ""
    total_fields: int = 0
    matched: int = 0
    mismatched: int = 0
    not_found: int = 0
    overall_verdict: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "INCONCLUSIVE"


class ReportGenerationInput(NodeBase):
    transaction_id: str
    coo_doc_id: str
    confirmed_template_id: str | None = None
    confirmed_template_name: str | None = None
    discrepancy_table: list[DiscrepancyItem]
    total_fields: int
    matched: int
    mismatched: int
    not_found: int


class ReportGenerationOutput(NodeBase):
    report: VerificationReport
    errors: list[str] = Field(default_factory=list)
