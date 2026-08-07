"""
Pydantic schemas for structured document extraction and cross-reference verification.

These schemas serve two purposes:
1. Used as ``with_structured_output(Schema)`` targets for LLM calls
2. Used as data models for storing/retrieving structured PACD/COO data

Architecture:
- PACD documents are stored as structured records (header_fields + line_items)
- COO documents are extracted into the same structured format
- Cross-reference verification operates on structured fields, not free-text chunks
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class _Base(BaseModel):
    model_config = {"strict": False, "extra": "forbid", "populate_by_name": True}


# ──────────────────────────────────────────────────────────────────────────────
# PACD Document Structuring — Vision LLM output schema
# ──────────────────────────────────────────────────────────────────────────────

class LineItem(_Base):
    """A single product/item line from a commercial document."""
    item_number: int | None = Field(None, description="Line number or position in the document (1-based)")
    hs_code: str | None = Field(None, description="Harmonized System code (e.g. '8471.30' or '8471.30.00')")
    description: str = Field(..., description="Product description text")
    quantity: str | None = Field(None, description="Quantity as stated (e.g. '500', '500 PCS')")
    unit: str | None = Field(None, description="Unit of measure (e.g. 'PCS', 'KG', 'MT', 'CTN')")
    unit_price: str | None = Field(None, description="Price per unit (e.g. '500.00', '500.00 USD')")
    total_value: str | None = Field(None, description="Total line value (e.g. '250000.00 USD')")
    weight: str | None = Field(None, description="Weight if stated (e.g. '2500 KG', '2.5 MT')")
    origin_country: str | None = Field(None, description="Country of origin for this specific item, if stated")
    raw_text: str | None = Field(None, description="Original unstructured text of this line item")


class PageStructuredExtraction(_Base):
    """Structured extraction output for a single document page.

    Used as ``with_structured_output(PageStructuredExtraction)`` target
    for the Vision LLM during PACD ingestion.
    """
    doc_type: str | None = Field(
        None,
        description=(
            "Type of document on this page. One of: "
            "'commercial_invoice', 'packing_list', 'bill_of_lading', "
            "'certificate_of_origin', 'insurance_certificate', 'contract', "
            "'other', or null if this is a continuation page with no new header."
        ),
    )
    is_continuation: bool = Field(
        False,
        description=(
            "True if this page is a continuation of the previous page's document "
            "(e.g. second page of an invoice with more line items but no new header)."
        ),
    )
    header_fields: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Header-level fields found on this page as key-value pairs. "
            "Use snake_case keys. Examples: exporter_name, consignee, invoice_number, "
            "date, port_of_loading, vessel_name, country_of_origin, etc. "
            "Only include fields that are clearly header/metadata, NOT line items."
        ),
    )
    line_items: list[LineItem] = Field(
        default_factory=list,
        description=(
            "Product/item lines found on this page. Each item should be a separate "
            "entry. If the page contains a table of products, extract each row as "
            "a separate LineItem. If no tabular items exist, leave empty."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# COO Extraction — structured output for COO documents
# ──────────────────────────────────────────────────────────────────────────────

class COOLineItem(_Base):
    """A single product line from a Certificate of Origin."""
    item_number: int | None = Field(None, description="Item/line number as shown on the COO")
    hs_code: str | None = Field(None, description="HS code / tariff code for this item")
    description: str = Field(..., description="Product/goods description")
    quantity: str | None = Field(None, description="Quantity (e.g. '500 PCS', '2000 KG')")
    unit: str | None = Field(None, description="Unit of measure")
    value: str | None = Field(None, description="Value/amount for this item")
    weight: str | None = Field(None, description="Weight (gross/net) if stated")
    origin_country: str | None = Field(None, description="Country of origin for this item")


class COOStructuredExtraction(_Base):
    """Complete structured extraction from a COO document.

    Used as ``with_structured_output(COOStructuredExtraction)`` target
    for the Vision LLM during COO extraction.
    """
    header_fields: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Header-level fields from the COO. Use snake_case keys. "
            "Expected fields include: exporter_name, consignee_name, "
            "country_of_origin, transport_details, port_of_loading, "
            "port_of_discharge, certificate_number, date_of_issue, "
            "issuing_authority, remarks, etc."
        ),
    )
    line_items: list[COOLineItem] = Field(
        default_factory=list,
        description=(
            "Product/goods items listed on the COO. Each distinct product "
            "with its own HS code or line number should be a separate entry."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Reference — LLM verification schemas (micro-batch)
# ──────────────────────────────────────────────────────────────────────────────

class HeaderFieldVerdict(_Base):
    """Verification verdict for a single header-level field."""
    field_key: str = Field(..., description="The header field name being verified")
    coo_value: str = Field(..., description="Value from the COO document")
    pacd_value: str | None = Field(None, description="Matching value found in PACD, or null")
    verdict: Literal["match", "mismatch", "not_found_in_pacd"] = Field(
        ..., description="match: values agree; mismatch: values differ; not_found_in_pacd: no evidence"
    )
    pacd_source_doc: str | None = Field(None, description="PACD document type where value was found")
    pacd_source_filename: str | None = Field(None, description="Filename of the PACD source")


class HeaderVerificationResult(_Base):
    """Structured output for header-level cross-reference verification.

    Used as ``with_structured_output(HeaderVerificationResult)`` target.
    """
    verdicts: list[HeaderFieldVerdict] = Field(
        ..., description="One verdict per COO header field. Every COO header field must appear exactly once."
    )


class ItemMatchCandidate(_Base):
    """A potential PACD line item match for a COO line item."""
    pacd_item_id: str = Field(..., description="ID of the PACD line item")
    description: str = Field(..., description="Description of the PACD item")
    hs_code: str | None = None
    confidence_reasoning: str = Field(..., description="Brief reason why this is or isn't a match")


class ItemMatchSelection(_Base):
    """LLM output for selecting the best PACD match for a COO item.

    Used as ``with_structured_output(ItemMatchSelection)`` target.
    """
    selected_pacd_item_id: str | None = Field(
        None,
        description="ID of the best matching PACD item, or null if none match"
    )
    reasoning: str = Field(..., description="Brief explanation of why this match was selected")


class ItemFieldVerdict(_Base):
    """Verification verdict for a single field within a line item comparison."""
    field_name: str = Field(..., description="Field being compared (e.g. 'hs_code', 'quantity', 'description')")
    coo_value: str | None = Field(None, description="Value from the COO item")
    pacd_value: str | None = Field(None, description="Value from the matched PACD item")
    verdict: Literal["match", "mismatch", "not_found_in_pacd"] = "not_found_in_pacd"


class ItemVerificationEntry(_Base):
    """Verification details for a single COO line item."""
    coo_item_number: int | None = Field(None, description="Item number from the COO")
    coo_description: str = Field(..., description="Description from the COO item")
    matched_pacd_item_id: str | None = Field(None, description="ID of matched PACD item, or null")
    matched_pacd_source: str | None = Field(None, description="Source filename of matched PACD item")
    field_verdicts: list[ItemFieldVerdict] = Field(
        default_factory=list,
        description="Per-field comparison verdicts for this item"
    )


class ItemVerificationResult(_Base):
    """Verification result for a batch of COO line items vs their matched PACD items.

    Used as ``with_structured_output(ItemVerificationResult)`` target.
    """
    items: list[ItemVerificationEntry] = Field(
        ..., description="One entry per COO line item in the batch"
    )


# ──────────────────────────────────────────────────────────────────────────────
# PACD Structuring Node — internal data models (not LLM output schemas)
# ──────────────────────────────────────────────────────────────────────────────

class LogicalDocument(_Base):
    """A logical sub-document assembled from one or more pages."""
    doc_type: str = Field(..., description="Document type classification")
    filename: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    header_fields: dict[str, str] = Field(default_factory=dict)
    line_items: list[LineItem] = Field(default_factory=list)
    raw_ocr_text: str = ""


class PACDStructuringOutput(_Base):
    """Output of the PACD structuring node."""
    transaction_id: str
    documents_created: int = 0
    line_items_created: int = 0
    errors: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Section-Based COO Verification — query generation + parallel verification
# ──────────────────────────────────────────────────────────────────────────────

class SectionQueries(_Base):
    """Retrieval queries generated from COO data for each PACD chunk section.

    Used as ``with_structured_output(SectionQueries)`` target for the
    query-generation LLM call before parallel verification.
    """
    header_queries: list[str] = Field(
        ...,
        description=(
            "3-5 concise search queries to retrieve PACD header chunks. "
            "Cover: exporter, consignee, invoice number, date, ports, country of origin."
        ),
    )
    content_queries: list[str] = Field(
        ...,
        description=(
            "3-5 concise search queries to retrieve PACD table/item chunks. "
            "Cover: HS codes, product descriptions, quantities, origin country."
        ),
    )
    footer_queries: list[str] = Field(
        ...,
        description=(
            "2-3 concise search queries to retrieve PACD footer chunks. "
            "Cover: totals, payment terms, remarks, bank details, authorized signatures."
        ),
    )


class FooterFieldVerdict(_Base):
    """Verification verdict for a single footer-level field."""
    field_key: str = Field(..., description="The footer field name being verified")
    coo_value: str = Field(..., description="Value from the COO document")
    pacd_value: str | None = Field(None, description="Matching value found in PACD, or null")
    verdict: Literal["match", "mismatch", "not_found_in_pacd"] = Field(
        ..., description="match: values agree; mismatch: values differ; not_found_in_pacd: no evidence"
    )
    pacd_source_doc: str | None = Field(None, description="PACD document type where value was found")


class FooterVerificationResult(_Base):
    """Structured output for footer-level cross-reference verification.

    Used as ``with_structured_output(FooterVerificationResult)`` target.
    """
    verdicts: list[FooterFieldVerdict] = Field(
        ...,
        description="One verdict per COO footer field. Every footer field must appear exactly once.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Map-Reduce RAG Pipeline — query expansion + LLM critic schemas
# ──────────────────────────────────────────────────────────────────────────────

class MultiQueryExpansion(_Base):
    """Multi-query expansion output for Map-Reduce RAG retrieval.

    Used as ``with_structured_output(MultiQueryExpansion)`` target in
    ``multi_query_expansion_node`` to generate targeted Milvus searches.
    """
    item_queries: list[str] = Field(
        ...,
        description=(
            "3-8 search queries for retrieving PACD table chunks that match the COO's "
            "line items. Include product names, synonyms, and HS code prefixes. "
            "Example: ['Stainless Steel Bolts', '7318.15', 'M8 Bolts fasteners']"
        ),
    )
    header_queries: list[str] = Field(
        ...,
        description=(
            "3-5 search queries for retrieving PACD header chunks. "
            "Cover: exporter/consignee names, invoice number, certificate number, date, ports. "
            "Use exact values from the COO (company names, reference numbers)."
        ),
    )
    footer_queries: list[str] = Field(
        ...,
        description=(
            "2-4 search queries for retrieving PACD footer chunks. "
            "Cover: total gross weight, total net weight, total value, payment terms. "
            "Use exact numeric values if present (e.g. 'total weight 500 KG')."
        ),
    )
    hs_queries: list[str] = Field(
        ...,
        description=(
            "All unique HS codes found in the COO line items, exactly as written. "
            "Example: ['7318.15', '8471.30']. Used for deterministic table-chunk matching."
        ),
    )


class CriticOutput(_Base):
    """Structured output from the single LLM Critic node.

    The LLM populates this from the condensed reference context.  The node
    then computes counts and assembles the final ``VerificationReport``.
    """
    discrepancy_table: list["DiscrepancyRow"] = Field(
        ...,
        description=(
            "One entry per COO field or line-item field verified against the PACD context. "
            "Cover every header field and every item field (hs_code, description, quantity, "
            "weight, value) for every COO line item."
        ),
    )
    summary: str = Field(..., description="2-3 sentence executive summary of the verification.")
    narrative: str = Field(
        ...,
        description=(
            "3-5 paragraph narrative describing findings, discrepancies, "
            "context gaps, and recommended follow-up actions."
        ),
    )
    overall_verdict: str = Field(
        ...,
        description="'PASS' if all verified fields match, 'FAIL' if any mismatch, 'INCONCLUSIVE' if evidence is insufficient.",
    )


class DiscrepancyRow(_Base):
    """A single field comparison row produced by the LLM critic."""
    field_key: str = Field(..., description="Field name, e.g. 'exporter_name' or 'item_1_hs_code'")
    coo_value: str | None = Field(None, description="Value from the COO document")
    pacd_value: str | None = Field(None, description="Matching value found in a PACD document")
    verdict: str = Field(
        ...,
        description="One of: 'match', 'mismatch', 'not_found_in_pacd'",
    )
    pacd_source_doc: str | None = Field(
        None, description="Document type where the PACD value was found (e.g. 'commercial_invoice')"
    )
