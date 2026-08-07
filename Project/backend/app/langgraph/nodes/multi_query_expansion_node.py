"""
MultiQueryExpansionNode — Reduce phase, step 1 of the Map-Reduce RAG pipeline.

Generates targeted Milvus retrieval queries from the extracted COO data.

Instead of a single generic search, this node produces four query categories
that map directly to specific Milvus chunk_type filters:

  item_queries   → searches "table" + "header" chunks (product line items)
  header_queries → searches "header" chunks only (exporter/consignee/refs)
  footer_queries → searches "footer" chunks only (totals/payment terms)
  hs_queries     → HS codes verbatim from COO items (deterministic match)

The Text LLM expands product names into synonyms and related terms so the
subsequent Milvus hybrid search catches items described differently across
PACD documents (e.g., "Stainless Steel Bolts" → "SS Bolts", "fasteners",
"7318.15").

Falls back to direct extraction from ``coo_extracted_data`` if the LLM call
fails, ensuring the pipeline never stalls on a query-expansion error.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.structured_extraction import MultiQueryExpansion
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

# Matches dotted HS codes: 8704.21, 8704.21.00
_HS_CODE_RE = re.compile(r'\b\d{2,4}\.\d{2,4}(?:\.\d{2,4})?\b')


def _normalize_hs_code(raw: str) -> str:
    """Convert '870421' → '8704.21', passthrough if already dotted."""
    raw = raw.strip().replace(' ', '').replace('-', '')
    if '.' in raw:
        return raw
    if raw.isdigit() and len(raw) == 6:
        return f"{raw[:4]}.{raw[4:]}"
    if raw.isdigit() and len(raw) == 8:
        return f"{raw[:4]}.{raw[4:6]}"
    return raw

_SYSTEM_PROMPT = """\
You are a trade document retrieval specialist.

Your task: given the structured fields extracted from a Certificate of Origin (COO),
generate targeted natural-language search queries for each of four categories.

CATEGORY 1 — item_queries
  Purpose: find PACD table chunks that contain matching line items.
  Rules: include the exact product name, common synonyms/abbreviations, and the HS code.
  Target count: 4-8 queries total covering all COO line items.
  Example: ["Stainless Steel Bolts M8", "7318.15", "SS Bolts fasteners", "Bolts hex head"]

CATEGORY 2 — header_queries
  Purpose: find PACD header chunks that contain exporter/consignee/reference info.
  Rules: use exact company names, invoice numbers, port names.
  Target count: 3-5 queries.
  Example: ["Acme Manufacturing Shanghai", "INV-2026-001", "Port of Shanghai Qingdao"]

CATEGORY 3 — footer_queries
  Purpose: find PACD footer chunks that contain total weights, values, payment terms.
  Rules: include exact numeric totals if present.
  Target count: 2-4 queries.
  Example: ["total gross weight 500 KG", "total net weight 450 KG", "total invoice value USD 50000"]

CATEGORY 4 — hs_queries
  Purpose: exact HS code lookup for deterministic table-chunk matching.
  Rules: output every unique HS code verbatim from the COO, NO expansion.
  Example: ["7318.15", "8471.30"]

Return ONLY the four lists. Do not add any explanations.
"""


def _extract_fallback_queries(coo_data: dict[str, Any]) -> dict[str, list[str]]:
    """Build queries deterministically from coo_extracted_data (no LLM)."""
    header_fields: dict[str, str] = coo_data.get("header_fields", {})
    line_items: list[dict] = coo_data.get("line_items", [])

    # item_queries: description of each item
    item_queries: list[str] = []
    hs_queries: list[str] = []
    for item in line_items:
        desc = (item.get("description") or "").strip()
        if desc:
            item_queries.append(desc)
            # Also scan description for embedded HS codes (e.g. "HS Code: 8704.21")
            for m in _HS_CODE_RE.findall(desc):
                norm = _normalize_hs_code(m)
                if norm not in hs_queries:
                    hs_queries.append(norm)
                    item_queries.append(norm)
        hs = (item.get("hs_code") or "").strip()
        if hs:
            norm = _normalize_hs_code(hs)
            if norm not in hs_queries:
                hs_queries.append(norm)
            if norm not in item_queries:
                item_queries.append(norm)
            # Include raw form too in case stored without dot
            if hs != norm and hs not in item_queries:
                item_queries.append(hs)

    # header_queries: key identifying header values
    header_queries: list[str] = []
    priority_keys = {"exporter_name", "consignee_name", "invoice_number",
                     "certificate_number", "port_of_loading", "country_of_origin"}
    for k, v in header_fields.items():
        if v and (k in priority_keys or len(header_queries) < 3):
            header_queries.append(v.strip())

    # footer_queries: total fields
    footer_queries: list[str] = ["total gross weight", "total net weight", "total value"]
    for k, v in header_fields.items():
        if "total" in k.lower() and v:
            footer_queries.append(f"{k}: {v}")

    return {
        "item":   item_queries[:8],
        "header": header_queries[:5],
        "footer": footer_queries[:4],
        "hs":     hs_queries,
    }


def _coo_summary(coo_data: dict[str, Any]) -> str:
    lines: list[str] = []
    header_fields: dict[str, str] = coo_data.get("header_fields", {})
    if header_fields:
        lines.append("=== COO Header Fields ===")
        for k, v in header_fields.items():
            lines.append(f"  {k}: {v}")
    line_items: list[dict] = coo_data.get("line_items", [])
    if line_items:
        lines.append(f"\n=== COO Line Items ({len(line_items)} items) ===")
        for item in line_items:
            parts = []
            if item.get("item_number"): parts.append(f"#{item['item_number']}")
            if item.get("hs_code"):     parts.append(f"HS:{item['hs_code']}")
            if item.get("description"): parts.append(item["description"])
            if item.get("quantity"):    parts.append(f"qty:{item['quantity']}")
            if item.get("weight"):      parts.append(f"wt:{item['weight']}")
            lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def multi_query_expansion_node(state: GraphState) -> GraphState:
    """LangGraph node: expand COO entities into multi-category search queries.

    Reads
    -----
    state["coo_extracted_data"]  — {"header_fields": {}, "line_items": [...]}

    Writes
    ------
    state["multi_queries"]   — {"item": [...], "header": [...], "footer": [...], "hs": [...]}
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("multi_query_expansion_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    coo_data: dict[str, Any] = state.get("coo_extracted_data") or {}

    if not coo_data:
        errors.append("multi_query_expansion_node: coo_extracted_data is empty — using fallback queries")
        return {
            **state,
            "multi_queries": _extract_fallback_queries({}),
            "errors": errors,
            "current_step": "multi_query_expansion_node",
        }  # type: ignore[return-value]

    summary = _coo_summary(coo_data)
    queries: dict[str, list[str]] | None = None

    try:
        llm = get_structured_llm(MultiQueryExpansion)
        response: MultiQueryExpansion = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"COO data:\n\n{summary}"),
        ])
        queries = {
            "item":   response.item_queries,
            "header": response.header_queries,
            "footer": response.footer_queries,
            # Normalise LLM-produced HS codes so "870421" → "8704.21"
            "hs":     [_normalize_hs_code(h) for h in response.hs_queries if h],
        }
        logger.info(
            "multi_query_expansion_node: generated %d item / %d header / %d footer / %d hs queries",
            len(queries["item"]), len(queries["header"]),
            len(queries["footer"]), len(queries["hs"]),
        )
    except Exception as exc:
        logger.warning(
            "multi_query_expansion_node: LLM call failed (%s) — falling back to direct extraction",
            exc,
        )
        errors.append(f"multi_query_expansion_node: LLM failed, used fallback — {exc}")
        queries = _extract_fallback_queries(coo_data)

    return {
        **state,
        "multi_queries": queries,
        "errors": errors,
        "current_step": "multi_query_expansion_node",
    }  # type: ignore[return-value]
