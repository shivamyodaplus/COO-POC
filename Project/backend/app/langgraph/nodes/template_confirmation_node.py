"""
TemplateConfirmationNode — asks the Vision LLM to confirm whether the COO
document matches each retrieved template candidate.

Strategy: sequential one-at-a-time
------------------------------------
Each LLM call compares the COO image against exactly ONE template image (2
images total per call).  Candidates are tried in descending similarity order
(as returned by Milvus).  The first candidate that the LLM confirms as a
match is accepted.  This keeps each call within the smallest possible image
budget, accommodating LLMs with strict image-count limits.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage

from app.langgraph.llm import encode_image_for_llm, get_vision_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    TemplateConfirmationInput,
    TemplateConfirmationOutput,
)
from app.langgraph.state import GraphState
from app.services.postgres_service import get_adapter
from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a document authentication expert.
You are given a scanned COO (Certificate of Origin) document image and ONE reference template image.

Determine whether the COO document was produced using this template.
Compare: overall layout, logos, field labels, table structure, borders, and general formatting.

Respond ONLY with a JSON object — no markdown, no prose:
{
  "is_match": true,
  "reasoning": "<brief explanation>"
}

Set "is_match" to false if the documents clearly differ in structure or branding.
"""

_SYSTEM_PROMPT_BATCH = """\
You are a document authentication expert.
You are given a scanned COO (Certificate of Origin) document image and several numbered reference \
template images.

Identify which template (if any) was used to produce the COO document.
Compare: overall layout, logos, field labels, table structure, borders, and general formatting.

Respond ONLY with a JSON object — no markdown, no prose:
{
  "confirmed_template_id": "<exact template id string, or null if none match>",
  "reasoning": "<brief explanation>"
}
"""


def template_confirmation_node(state: GraphState) -> GraphState:
    """LangGraph node: Vision LLM confirms which template the COO matches.

    Tries each retrieved candidate one-by-one (2 images per LLM call: COO +
    template) in descending similarity order.  Stops at the first confirmed
    match.  This keeps every call within a minimal image budget regardless of
    the LLM's image-count limit.

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

    adapter = get_adapter()
    llm = get_vision_llm(temperature=0.0)
    batch_size = max(1, settings.COO_TEMPLATE_CONFIRM_BATCH)

    confirmed_template_id: str | None = None
    confirmed_template_name: str | None = None

    # Load images for all candidates up-front so we can skip ones without images
    loaded: list[tuple[dict, object]] = []   # (candidate, img)
    for candidate in node_input.candidates:
        template_id = candidate.get("id") or candidate.get("template_id")
        if not template_id:
            continue
        try:
            img = adapter.get_visual_template_image(str(template_id))
        except Exception as exc:
            logger.warning("template_confirmation_node: could not load image for %s — %s", template_id, exc)
            continue
        if img is None:
            logger.warning("template_confirmation_node: no image stored for template %s", template_id)
            continue
        loaded.append((candidate, img))

    if not loaded:
        errors.append("template_confirmation_node: could not load any template images from storage")
        return {**state, "errors": errors, "current_step": "template_confirmation_node"}  # type: ignore[return-value]

    # Iterate in batches of batch_size (order = descending Milvus similarity)
    for batch_start in range(0, len(loaded), batch_size):
        batch = loaded[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(loaded) + batch_size - 1) // batch_size
        logger.info(
            "template_confirmation_node: batch %d/%d (%d template(s))",
            batch_num, total_batches, len(batch),
        )

        if batch_size == 1:
            # ── Single-template call: binary is_match ──────────────────────
            candidate, img = batch[0]
            template_id = str(candidate.get("id") or candidate.get("template_id"))
            template_name = candidate.get("name", template_id)

            content_parts: list[dict] = [
                {"type": "text", "text": _SYSTEM_PROMPT},
                {"type": "text", "text": "COO Document (to match):"},
                {"type": "image_url", "image_url": {"url": node_input.coo_image_b64}},
                {"type": "text", "text": f"Reference Template (id={template_id}, name={template_name}):"},
                {"type": "image_url", "image_url": {"url": encode_image_for_llm(img.data, img.content_type)}},  # type: ignore[union-attr]
            ]

            try:
                response = llm.invoke([HumanMessage(content=content_parts)])
                raw = str(response.content).strip()
                if "<think>" in raw:
                    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(raw)
                is_match = bool(parsed.get("is_match", False))
                logger.info(
                    "template_confirmation_node: id=%s is_match=%s — %s",
                    template_id, is_match, parsed.get("reasoning", ""),
                )
                if is_match:
                    confirmed_template_id = template_id
                    confirmed_template_name = template_name
                    break
            except Exception as exc:
                errors.append(
                    f"template_confirmation_node: LLM call failed for batch {batch_num} ({template_id}) — {exc}"
                )
                logger.warning("template_confirmation_node: skipping batch %d: %s", batch_num, exc)

        else:
            # ── Multi-template call: LLM picks the matching id ─────────────
            content_parts = [
                {"type": "text", "text": _SYSTEM_PROMPT_BATCH},
                {"type": "text", "text": "COO Document (to match):"},
                {"type": "image_url", "image_url": {"url": node_input.coo_image_b64}},
            ]
            id_to_candidate: dict[str, dict] = {}
            for idx, (candidate, img) in enumerate(batch, start=1):
                t_id = str(candidate.get("id") or candidate.get("template_id"))
                t_name = candidate.get("name", t_id)
                content_parts.append({"type": "text", "text": f"Template {idx} (id={t_id}, name={t_name}):"})
                content_parts.append({"type": "image_url", "image_url": {"url": encode_image_for_llm(img.data, img.content_type)}})  # type: ignore[union-attr]
                id_to_candidate[t_id] = candidate

            try:
                response = llm.invoke([HumanMessage(content=content_parts)])
                raw = str(response.content).strip()
                if "<think>" in raw:
                    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(raw)
                matched_id: str | None = parsed.get("confirmed_template_id") or None
                logger.info(
                    "template_confirmation_node: batch %d confirmed_template_id=%s — %s",
                    batch_num, matched_id, parsed.get("reasoning", ""),
                )
                if matched_id and matched_id in id_to_candidate:
                    confirmed_template_id = matched_id
                    confirmed_template_name = id_to_candidate[matched_id].get("name", matched_id)
                    break
            except Exception as exc:
                errors.append(
                    f"template_confirmation_node: LLM call failed for batch {batch_num} — {exc}"
                )
                logger.warning("template_confirmation_node: skipping batch %d: %s", batch_num, exc)

    # Load extracted_attributes for the confirmed template
    template_attributes: dict = {}
    if confirmed_template_id:
        try:
            pool = __import__("app.services.postgres_service", fromlist=["get_pool"]).get_pool()
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT extracted_attributes FROM visual_templates WHERE id = %s",
                    (confirmed_template_id,),
                )
                row = cur.fetchone()
            if row and row[0]:
                attrs = row[0]
                template_attributes = attrs if isinstance(attrs, dict) else json.loads(attrs)
        except Exception as exc:
            errors.append(f"template_confirmation_node: failed to load template attributes — {exc}")

    if not confirmed_template_id:
        logger.info(
            "template_confirmation_node: no template confirmed after checking %d candidates",
            len(node_input.candidates),
        )
    else:
        logger.info(
            "template_confirmation_node: confirmed=%s (%s)",
            confirmed_template_id, confirmed_template_name,
        )

    return {  # type: ignore[return-value]
        **state,
        "confirmed_template_id": confirmed_template_id,
        "confirmed_template_name": confirmed_template_name,
        "template_attributes": template_attributes,
        "errors": errors,
        "current_step": "template_confirmation_node",
    }
