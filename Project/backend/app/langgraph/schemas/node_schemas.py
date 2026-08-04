"""
Pydantic schemas for every node's input and output contract.

Design rules
------------
- Every node must declare an ``Input`` and an ``Output`` model in this file.
- Models inherit from ``NodeBase`` which enables strict validation and
  forbids extra fields, ensuring nodes never silently accept garbage data.
- Field descriptions act as inline documentation *and* are forwarded to the
  LLM as tool/schema context when using structured output.
- All optional fields default to ``None``; mandatory fields have no default.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #

class NodeBase(BaseModel):
    """Shared configuration inherited by all node schemas."""

    model_config = {
        "strict": True,          # values must match declared types exactly
        "extra": "forbid",        # reject unknown keys
        "populate_by_name": True, # allow alias OR field name when constructing
    }


# --------------------------------------------------------------------------- #
# IngestionNode                                                                #
# --------------------------------------------------------------------------- #

class IngestionNodeInput(NodeBase):
    """Data required to start the ingestion pipeline."""

    raw_bytes: bytes = Field(..., description="Raw file bytes of the uploaded document.")
    filename: str = Field(..., description="Original filename including extension.")
    country: str | None = Field(None, description="User-supplied country hint (ISO 3166-1 alpha-2).")
    doc_type: str | None = Field(None, description="User-supplied document type hint.")


class IngestionNodeOutput(NodeBase):
    """Data produced by the ingestion node."""

    document_id: str = Field(..., description="Unique identifier assigned to this document.")
    mime_type: str = Field(..., description="Detected MIME type of the uploaded file.")
    page_count: int = Field(..., description="Number of pages extracted from the document.")
    errors: list[str] = Field(default_factory=list, description="Non-fatal errors encountered during ingestion.")


# --------------------------------------------------------------------------- #
# ClassificationNode                                                           #
# --------------------------------------------------------------------------- #

class ClassificationNodeInput(NodeBase):
    """Data required for the classification node."""

    document_id: str = Field(..., description="Document identifier from the ingestion step.")
    ocr_text: str = Field(..., description="Full OCR text extracted from all pages.")
    country_hint: str | None = Field(None, description="Optional country hint to narrow classification.")


class ClassificationNodeOutput(NodeBase):
    """Data produced by the classification node."""

    country: str = Field(..., description="Predicted country (ISO 3166-1 alpha-2).")
    doc_type: str = Field(..., description="Predicted document type (e.g. passport, invoice).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score [0, 1].")
    reasoning: str | None = Field(None, description="LLM chain-of-thought or brief explanation.")


# --------------------------------------------------------------------------- #
# ExtractionNode                                                               #
# --------------------------------------------------------------------------- #

class ExtractedField(NodeBase):
    """A single key-value pair extracted from the document."""

    key: str = Field(..., description="Field name (e.g. 'full_name', 'date_of_birth').")
    value: str = Field(..., description="Extracted value as a string.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Per-field extraction confidence.")


class ExtractionNodeInput(NodeBase):
    """Data required for the extraction node."""

    document_id: str = Field(..., description="Document identifier.")
    doc_type: str = Field(..., description="Resolved document type used to select the extraction prompt.")
    ocr_text: str = Field(..., description="Full OCR text to extract fields from.")


class ExtractionNodeOutput(NodeBase):
    """Data produced by the extraction node."""

    extracted_fields: list[ExtractedField] = Field(
        default_factory=list,
        description="All key-value fields extracted from the document.",
    )
    raw_llm_response: str | None = Field(
        None,
        description="Unprocessed LLM response for debugging / audit purposes.",
    )


# --------------------------------------------------------------------------- #
# WorkflowInput / WorkflowOutput  (top-level API boundary schemas)            #
# --------------------------------------------------------------------------- #

class WorkflowInput(NodeBase):
    """Public input schema for invoking any workflow via the adapter."""

    raw_bytes: bytes = Field(..., description="Raw file bytes of the document to process.")
    filename: str = Field(..., description="Original filename.")
    country: str | None = Field(None, description="Optional country hint.")
    doc_type: str | None = Field(None, description="Optional document-type hint.")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-specific extra parameters.",
    )


class WorkflowOutput(NodeBase):
    """Public output schema returned by any workflow via the adapter."""

    document_id: str | None = Field(None, description="Assigned document ID.")
    country: str | None = Field(None, description="Detected country.")
    doc_type: str | None = Field(None, description="Detected document type.")
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Flat dict of extracted key-value pairs.",
    )
    errors: list[str] = Field(default_factory=list, description="Accumulated errors.")
