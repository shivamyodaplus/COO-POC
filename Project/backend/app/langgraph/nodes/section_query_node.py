"""
SectionQueryNode — generates targeted Milvus retrieval queries for each
document section (header / table / footer) from the extracted COO data.

This node sits between coo_extraction_node and the three parallel
verification nodes.  Its single LLM call produces SectionQueries, which
the downstream nodes use to retrieve the most relevant PACD chunks for
their respective sections.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.structured_extraction import SectionQueries
from app.langgraph.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a trade document retrieval specialist.

Given the extracted fields from a Certificate of Origin (COO) document, your task is to
generate concise natural-language search queries for retrieving relevant sections from
Pre-Arrival Customs Documents (PACD) stored in a vector database.

Generate queries for THREE sections:

1. HEADER queries — to retrieve PACD header chunks containing:
   exporter details, consignee details, invoice numbers, issue dates,
   ports of loading/discharge, country of origin, transport info.

2. CONTENT queries — to retrieve PACD table/item chunks containing:
   HS codes, product descriptions, quantities, unit prices, weights,
   country of origin for goods, line item details.

3. FOOTER queries — to retrieve PACD footer chunks containing:
   total amounts, gross/net weights, payment terms, bank details,
   remarks, authorized signatures, certifications.

Rules:
- Each query should be a short phrase (5-15 words) — not a sentence.
- Include key values from the COO (names, numbers, codes) in the queries.
- Generate 3-5 header queries, 3-5 content queries, 2-3 footer queries.
- Prioritize specificity: exact values (invoice numbers, HS codes, company names)
  make better queries than generic terms.
"""


def _build_coo_summary(coo_data: dict[str, Any]) -> str:
    """Produce a compact text summary of COO extracted data for the prompt."""
    lines: list[str] = []

    header_fields: dict[str, str] = coo_data.get("header_fields", {})
    if header_fields:
        lines.append("COO Header Fields:")
        for k, v in header_fields.items():
            lines.append(f"  {k}: {v}")

    line_items: list[dict[str, Any]] = coo_data.get("line_items", [])
    if line_items:
        lines.append("\nCOO Line Items:")
        for i, item in enumerate(line_items[:10], start=1):  # cap at 10 items
            parts = []
            if item.get("hs_code"):        parts.append(f"HS {item['hs_code']}")
            if item.get("description"):    parts.append(item["description"])
            if item.get("quantity"):       parts.append(f"qty {item['quantity']}")
            if item.get("weight"):         parts.append(f"weight {item['weight']}")
            if item.get("origin_country"): parts.append(f"origin {item['origin_country']}")
            lines.append(f"  Item {i}: {' | '.join(parts)}")

    return "\n".join(lines) if lines else "No COO data available."


def section_query_node(state: GraphState) -> GraphState:
    """LangGraph node: generate section retrieval queries from COO data.

    Reads
    -----
    state["coo_extracted_data"]   — {"header_fields": {}, "line_items": [...]}

    Writes
    ------
    state["section_queries"]  — {"header": [...], "content": [...], "footer": [...]}
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("section_query_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    coo_data: dict[str, Any] = state.get("coo_extracted_data") or {}

    if not coo_data:
        errors.append("section_query_node: coo_extracted_data is empty — using minimal fallback queries")
        return {  # type: ignore[return-value]
            **state,
            "section_queries": {
                "header": ["exporter consignee invoice"],
                "content": ["product HS code quantity"],
                "footer": ["total weight payment terms"],
            },
            "errors": errors,
            "current_step": "section_query_node",
        }

    coo_summary = _build_coo_summary(coo_data)

    llm = get_structured_llm(SectionQueries, temperature=0.0)
    try:
        result: SectionQueries = llm.invoke([  # type: ignore[assignment]
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"COO Document Data:\n{coo_summary}"),
        ])
        section_queries = {
            "header":  result.header_queries,
            "content": result.content_queries,
            "footer":  result.footer_queries,
        }
        logger.info(
            "section_query_node: generated %d header / %d content / %d footer queries",
            len(result.header_queries),
            len(result.content_queries),
            len(result.footer_queries),
        )
    except Exception as exc:
        logger.warning("section_query_node: LLM failed — %s, using value-based fallback", exc)
        errors.append(f"section_query_node: query generation failed — {exc}")
        # Fallback: extract key values directly from COO data
        header_fields = coo_data.get("header_fields", {})
        items = coo_data.get("line_items", [])
        section_queries = {
            "header": [v for v in list(header_fields.values())[:5] if v],
            "content": [
                f"{item.get('hs_code', '')} {item.get('description', '')}".strip()
                for item in items[:4] if item.get("description")
            ] or ["product description HS code"],
            "footer": ["total amount weight", "payment terms remarks"],
        }

    return {  # type: ignore[return-value]
        **state,
        "section_queries": section_queries,
        "errors": errors,
        "current_step": "section_query_node",
    }
