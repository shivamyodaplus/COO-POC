"""
TemplateConfirmationNode — asks the Vision LLM to compare the COO document
image against the top-k retrieved template candidates and confirm which one
the user has actually used.

The node also fetches each candidate's stored image bytes from Postgres for
visual comparison, and loads the ``extracted_attributes`` field to carry
forward into the extraction step.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from app.langgraph.llm import encode_image_for_llm, get_vision_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    TemplateConfirmationInput,
    TemplateConfirmationOutput,
)
from app.langgraph.state import GraphState
from app.services.postgres_service import get_adapter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a document authentication expert.
You are given:
  1. A scanned COO (Certificate of Origin) document image.
  2. Up to 3 reference template images that the issuer might have used.

Your task: identify which template (if any) was used to produce the COO document.
Compare layout, logos, field positions, borders, and formatting.

Respond ONLY with a JSON object:
{
  "confirmed_template_id": "<template_id or null>",
  "confirmed_template_name": "<template_name or null>",
  "reasoning": "<brief explanation>"
}

If none of the templates match, set confirmed_template_id to null.
"""


def template_confirmation_node(state: GraphState) -> GraphState:
    """LangGraph node: Vision LLM confirms which template the COO matches.

    Reads
    -----
    state["page_images"]         — COO first page image
    state["retrieved_templates"] — candidate template records [{id, name, ...}]

    Writes
    ------
    state["confirmed_template_id"], state["confirmed_template_name"],
    state["template_attributes"] (loaded from confirmed template's DB record),
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("template_confirmation_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    page_images: list[bytes] = state.get("page_images") or []
    candidates: list[dict] = state.get("retrieved_templates") or []

    if not page_images:
        errors.append("template_confirmation_node: no page_images in state")
        return {**state, "errors": errors, "current_step": "template_confirmation_node"}  # type: ignore[return-value]

    if not candidates:
        errors.append("template_confirmation_node: no template candidates to compare")
        return {**state, "errors": errors, "current_step": "template_confirmation_node"}  # type: ignore[return-value]

    try:
        node_input = TemplateConfirmationInput(
            document_id=state.get("document_id") or "",
            coo_image_b64=encode_image_for_llm(page_images[0], "image/jpeg"),
            candidates=candidates,
        )
    except Exception as exc:
        errors.append(f"template_confirmation_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "template_confirmation_node"}  # type: ignore[return-value]

    # Fetch template images from storage
    adapter = get_adapter()
    content_parts: list[dict] = [
        {"type": "text", "text": _SYSTEM_PROMPT},
        {"type": "text", "text": "COO Document (to match):"},
        {"type": "image_url", "image_url": {"url": node_input.coo_image_b64}},
    ]

    valid_candidates: list[dict] = []
    for idx, candidate in enumerate(node_input.candidates[:3], start=1):
        template_id = candidate.get("id") or candidate.get("template_id")
        if not template_id:
            continue
        try:
            img = adapter.get_visual_template_image(str(template_id))
        except Exception:
            continue
        if img is None:
            continue
        content_parts.append({"type": "text", "text": f"Template {idx}: id={template_id}, name={candidate.get('name', '')}"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": encode_image_for_llm(img.data, img.content_type)},
        })
        valid_candidates.append(candidate)

    if not valid_candidates:
        errors.append("template_confirmation_node: could not load any template images from storage")
        return {**state, "errors": errors, "current_step": "template_confirmation_node"}  # type: ignore[return-value]

    llm = get_vision_llm(temperature=0.0)
    raw_response = ""
    output = TemplateConfirmationOutput()

    try:
        response = llm.invoke([HumanMessage(content=content_parts)])
        raw_response = str(response.content)
        parsed = json.loads(raw_response)
        output = TemplateConfirmationOutput(**parsed, errors=errors)
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"template_confirmation_node: bad LLM JSON — {exc}: {raw_response[:200]}")
    except Exception as exc:
        errors.append(f"template_confirmation_node: LLM call failed — {exc}")

    output.errors = errors

    # Load extracted_attributes for the confirmed template
    template_attributes: dict = {}
    if output.confirmed_template_id:
        try:
            # Try to get from the adapter's metadata; attributes are in extracted_attributes JSONB
            pool = __import__("app.services.postgres_service", fromlist=["get_pool"]).get_pool()
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT extracted_attributes FROM visual_templates WHERE id = %s",
                    (output.confirmed_template_id,),
                )
                row = cur.fetchone()
            if row and row[0]:
                attrs = row[0]
                template_attributes = attrs if isinstance(attrs, dict) else json.loads(attrs)
        except Exception as exc:
            errors.append(f"template_confirmation_node: failed to load template attributes — {exc}")

    logger.info(
        "template_confirmation_node: confirmed=%s (%s)",
        output.confirmed_template_id,
        output.confirmed_template_name,
    )

    return {  # type: ignore[return-value]
        **state,
        "confirmed_template_id": output.confirmed_template_id,
        "confirmed_template_name": output.confirmed_template_name,
        "template_attributes": template_attributes,
        "errors": errors,
        "current_step": "template_confirmation_node",
    }
