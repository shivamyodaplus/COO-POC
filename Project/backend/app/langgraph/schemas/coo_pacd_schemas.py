"""
Pydantic schemas for template retrieval, confirmation, and attribute extraction.

These schemas define the input/output contracts for the template-matching
nodes in the COO verification workflow.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #

class NodeBase(BaseModel):
    model_config = {
        "strict": False,
        "extra": "forbid",
        "populate_by_name": True,
    }


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
