"""
Unified graph state shared across every node in every workflow.

Rules
-----
- This is the single source of truth for what flows through the graph.
- Every field must have an explicit type annotation.
- `messages` uses LangGraph's ``add_messages`` reducer so that node outputs
  are *appended* rather than overwriting the list.
- All other fields are overwrite-on-assignment (last writer wins).
- Add new fields here when a new node needs to pass data downstream; never
  pass data outside the state dict.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    # ------------------------------------------------------------------ #
    # Conversation / LLM messages                                          #
    # ------------------------------------------------------------------ #
    # add_messages reducer appends new messages rather than replacing them.
    messages: Annotated[list, add_messages]

    # ------------------------------------------------------------------ #
    # Identity & transaction scope                                         #
    # ------------------------------------------------------------------ #
    user_id: str | None              # caller identity (plain string, no auth for POC)
    transaction_id: str | None       # transaction this document belongs to
    doc_category: str | None         # 'pacd' | 'coo'

    # ------------------------------------------------------------------ #
    # Document context                                                     #
    # ------------------------------------------------------------------ #
    document_id: str | None          # unique ID assigned after ingestion
    raw_bytes: bytes | None          # raw file bytes from the upload
    filename: str | None             # original filename
    mime_type: str | None            # detected MIME / content-type
    page_images: list[bytes]         # per-page JPEG bytes (from ingestion)

    # ------------------------------------------------------------------ #
    # Extracted / classified information (generic)                         #
    # ------------------------------------------------------------------ #
    country: str | None              # detected or user-supplied country
    doc_type: str | None             # document type hint
    extracted_fields: dict[str, Any] # structured key-value pairs (generic use)
    extracted_kv_pairs: list[dict[str, Any]]  # per-page [{page, key, value, confidence}]

    # ------------------------------------------------------------------ #
    # Template matching                                                    #
    # ------------------------------------------------------------------ #
    retrieved_templates: list[dict[str, Any]]  # top-k Milvus hits
    template_attributes: dict[str, Any]        # {attr_key: description} from template
    confirmed_template_id: str | None          # winner after Vision LLM confirmation
    confirmed_template_name: str | None

    # ------------------------------------------------------------------ #
    # COO verification pipeline                                            #
    # ------------------------------------------------------------------ #
    coo_extracted_fields: dict[str, Any]       # fields extracted from COO document
    cross_reference_results: list[dict[str, Any]]  # list of DiscrepancyItem dicts
    verification_report: dict[str, Any]        # final report {table, narrative, summary}

    # ------------------------------------------------------------------ #
    # Control flow                                                         #
    # ------------------------------------------------------------------ #
    errors: list[str]                # accumulated non-fatal errors
    current_step: str | None         # last completed node name (for debugging)
