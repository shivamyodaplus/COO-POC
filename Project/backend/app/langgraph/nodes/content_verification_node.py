"""
ContentVerificationNode — verifies COO line items against retrieved PACD
table (body) section chunks.

Runs in parallel with header_verification_node and footer_verification_node.

Flow
----
1. Read content_queries from state["section_queries"]["content"].
2. Search Milvus pacd_chunks (chunk_type="table") using RRF-fused multi-query search.
3. Build context: COO line items + retrieved PACD table chunk texts.
4. Single LLM call → ItemVerificationResult (per-item, per-field verdicts).
5. Write results to state["content_verification_results"].
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.structured_extraction import ItemVerificationResult
from app.langgraph.state import GraphState
from app.services import pacd_milvus_service

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a trade compliance expert verifying line items from a Certificate of Origin (COO)
against Pre-Arrival Customs Documents (PACD).

You are given:
1. COO line items — products/goods listed on the certificate.
2. Retrieved PACD table chunks — item tables from PACD documents for this transaction.

For EACH COO line item:
  a. Find the best matching entry in the PACD table chunks.
  b. Compare the following fields: hs_code, description, quantity, weight, origin_country.

Verdict rules per field:
- "match": Values agree (case-insensitive; minor formatting ok; "500 KG" = "500 Kilograms").
- "mismatch": Values differ materially.
- "not_found_in_pacd": Field or item not found in any PACD chunk.

HS code matching: "8471.30" matches "8471.30.00" (prefix match is sufficient).

Set matched_pacd_item_id to a descriptive label (e.g. "PACD Item 1 from invoice") when found,
or null when no match exists.
Set matched_pacd_source to the doc_type of the source chunk.

Every COO line item must appear exactly once in the output.
"""


def _format_coo_items(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        parts = [f"COO Item {item.get('item_number', i)}:"]
        if item.get("hs_code"):        parts.append(f"  HS Code: {item['hs_code']}")
        if item.get("description"):    parts.append(f"  Description: {item['description']}")
        if item.get("quantity"):       parts.append(f"  Quantity: {item['quantity']}")
        if item.get("weight"):         parts.append(f"  Weight: {item['weight']}")
        if item.get("value"):          parts.append(f"  Value: {item['value']}")
        if item.get("origin_country"): parts.append(f"  Origin: {item['origin_country']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines) if lines else "No line items in COO."


def _format_pacd_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No PACD table chunks retrieved."
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta_items: list[dict] = []
        try:
            meta = json.loads(chunk.get("chunk_metadata", "{}"))
            meta_items = meta.get("line_items", [])
        except Exception:
            pass

        header = (
            f"[Chunk {i} | {chunk.get('doc_type', '?')} | score={chunk.get('rrf_score', 0):.4f}]\n"
            f"{chunk.get('chunk_text', '')}"
        )
        if meta_items:
            header += f"\n  (metadata: {len(meta_items)} items)"
        lines.append(header)
    return "\n\n".join(lines)


def content_verification_node(state: GraphState) -> GraphState:
    """LangGraph node: verify COO line items against retrieved PACD table chunks.

    Reads
    -----
    state["section_queries"]["content"]
    state["coo_extracted_data"]["line_items"]
    state["transaction_id"]

    Writes
    ------
    state["content_verification_results"]  — list of ItemVerificationEntry dicts
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("content_verification_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    transaction_id: str = state.get("transaction_id") or ""
    coo_data: dict[str, Any] = state.get("coo_extracted_data") or {}
    coo_items: list[dict[str, Any]] = coo_data.get("line_items", [])
    section_queries: dict[str, Any] = state.get("section_queries") or {}
    content_queries: list[str] = section_queries.get("content", [])

    if not coo_items:
        logger.info("content_verification_node: no COO line items to verify")
        return {"content_verification_results": []}  # type: ignore[return-value]

    # ── Retrieve PACD table chunks ─────────────────────────────────────────
    chunks: list[dict[str, Any]] = []
    if content_queries and transaction_id:
        try:
            chunks = pacd_milvus_service.search_section_chunks(
                section="table",
                queries=content_queries,
                transaction_id=transaction_id,
                top_k=10,
                top_t=8,
            )
            logger.info("content_verification_node: retrieved %d table chunks", len(chunks))
        except Exception as exc:
            errors.append(f"content_verification_node: Milvus search failed — {exc}")
            logger.warning("content_verification_node: Milvus error: %s", exc)

    # ── Build LLM context ──────────────────────────────────────────────────
    coo_block = _format_coo_items(coo_items)
    pacd_block = _format_pacd_chunks(chunks)

    user_msg = (
        f"COO Line Items:\n{coo_block}\n\n"
        f"Retrieved PACD Table Chunks:\n{pacd_block}"
    )

    # ── LLM verification call ──────────────────────────────────────────────
    llm = get_structured_llm(ItemVerificationResult, temperature=0.0)
    try:
        result: ItemVerificationResult = llm.invoke([  # type: ignore[assignment]
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        item_results = [entry.model_dump() for entry in result.items]
        logger.info(
            "content_verification_node: %d item verification entries produced",
            len(item_results),
        )
    except Exception as exc:
        logger.warning("content_verification_node: LLM error: %s", exc)
        # Fallback: mark all items as not found
        item_results = [
            {
                "coo_item_number":       item.get("item_number", i + 1),
                "coo_description":       item.get("description", ""),
                "matched_pacd_item_id":  None,
                "matched_pacd_source":   None,
                "field_verdicts": [
                    {"field_name": f, "coo_value": str(item.get(f, "")),
                     "pacd_value": None, "verdict": "not_found_in_pacd"}
                    for f in ["hs_code", "description", "quantity", "weight"]
                    if item.get(f)
                ],
            }
            for i, item in enumerate(coo_items)
        ]

    return {"content_verification_results": item_results}  # type: ignore[return-value]
