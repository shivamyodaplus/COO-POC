"""
COO Transcription Node — Phase 2a of 2-Tier GraphRAG.

Uses Vision LLM to:
  Step 1: Transcribe all visible text from the COO page images
  Step 2: Generate up to MAX_VALIDATION_QUERIES targeted verification queries
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.langgraph.llm import encode_image_for_llm, get_vision_llm
from app.langgraph.schemas.graphrag_schemas import ValidationQueryList
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)


def coo_transcription_node(state: GraphState) -> dict[str, Any]:
    """Transcribe COO text and generate validation queries via Vision LLM."""
    page_images: list[bytes] = state.get("page_images") or []
    transaction_id: str = state.get("transaction_id") or ""

    if not page_images:
        logger.warning("coo_transcription_node: no page images")
        return {
            "coo_text": "",
            "validation_queries": [],
            "errors": state.get("errors", []) + ["No COO page images available"],
            "current_step": "coo_transcription_node",
        }

    # Encode all pages as base64 data URIs
    image_contents: list[dict[str, Any]] = []
    for img_bytes in page_images:
        data_uri = encode_image_for_llm(img_bytes)
        image_contents.append({"type": "image_url", "image_url": {"url": data_uri}})

    llm = get_vision_llm(temperature=0.0)

    # ── Step 1: Transcribe COO text ──────────────────────────────────────────
    transcription_content = list(image_contents) + [{
        "type": "text",
        "text": (
            "You are a document transcription assistant. "
            "Transcribe ALL text from this Certificate of Origin document exactly as it appears — "
            "including field labels, values, dates, codes, and signatures. "
            "Preserve the original structure using newlines. Return only the raw transcription, no commentary."
        ),
    }]

    try:
        response = llm.invoke([HumanMessage(content=transcription_content)])
        coo_text = response.content.strip()
    except Exception as exc:
        logger.error("coo_transcription_node: transcription failed — %s", exc)
        return {
            "coo_text": "",
            "validation_queries": [],
            "errors": state.get("errors", []) + [f"COO transcription failed: {exc}"],
            "current_step": "coo_transcription_node",
        }

    logger.info("coo_transcription_node: transcribed %d chars", len(coo_text))

    # ── Step 2: Generate validation queries ──────────────────────────────────
    max_queries = settings.MAX_VALIDATION_QUERIES
    schema_json = json.dumps(ValidationQueryList.model_json_schema(), indent=2)

    query_prompt = f"""You are a trade compliance auditor. Look at this Certificate of Origin (COO) document and generate up to {max_queries} specific, concise verification queries that can be answered by searching the reference trade documents (invoice, packing list).

Each query should:
- Target one specific field (item name, HS code, quantity, net weight, gross weight, unit price, total value, origin country, exporter name, etc.)
- Be phrased as a direct factual question (e.g. "What is the HS code for cocoa beans?")
- Be independently searchable

Respond ONLY with a JSON object matching this schema:
{schema_json}"""

    query_content = list(image_contents) + [{"type": "text", "text": query_prompt}]

    system_text = (
        "You are a precise data extraction assistant. "
        "You MUST respond with valid JSON that matches the schema exactly. "
        "Return ONLY the JSON object, no markdown fences, no extra text."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=query_content),
        ])
        raw_json = response.content.strip()

        # Handle markdown fences if present
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw_json)
        query_list = ValidationQueryList(**parsed)
        validation_queries = query_list.queries[:max_queries]

    except Exception as exc:
        logger.warning("coo_transcription_node: query generation failed — %s", exc)
        # Fallback: extract basic queries from transcribed text
        validation_queries = _fallback_queries(coo_text)

    logger.info(
        "coo_transcription_node: generated %d validation queries", len(validation_queries)
    )

    return {
        "coo_text": coo_text,
        "validation_queries": validation_queries,
        "errors": state.get("errors", []),
        "current_step": "coo_transcription_node",
    }


def _fallback_queries(coo_text: str) -> list[str]:
    """Generate basic queries from COO text when VLM query generation fails."""
    queries = []
    # Extract lines that look like field values
    for line in coo_text.split("\n"):
        line = line.strip()
        if ":" in line and len(line) > 10 and len(line) < 200:
            key = line.split(":", 1)[0].strip()
            if key and len(key) < 60:
                queries.append(f"What is the {key}?")
        if len(queries) >= 10:
            break
    return queries
