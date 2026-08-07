"""
CrossReferenceNode (v2) — structured 3-pass verification of COO against PACD.

Architecture: Micro-batch verification to minimize hallucination.
-----------------------------------------------------------------
Instead of dumping all evidence into one large LLM call, we split verification
into small, focused calls with minimal context per call.

Pass 1: Header Field Verification
    - SQL: fetch all PACD document headers for this transaction
    - LLM: single call comparing COO headers vs PACD headers (~500 tokens)
    - Output: per-header-field verdict

Pass 2: Line Item Matching
    - SQL: exact HS code lookup in pacd_line_items
    - Milvus fallback: semantic search on description if HS match fails
    - LLM (optional): disambiguation when multiple candidates exist (~100 tokens)
    - Output: COO item → matched PACD item mapping

Pass 3: Per-Item Field Verification
    - LLM: batch 3-5 items per call, comparing COO item fields vs matched PACD item
    - Context per call: ~600-1000 tokens max
    - Output: per-field verdict for each item

Total LLM calls: ~3-8 (vs old approach: 1 large unreliable call)
Max tokens per call: ~1000 (vs old approach: 3000-5000 in single call)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.langgraph.llm import get_structured_llm
from app.langgraph.schemas.coo_pacd_schemas import DiscrepancyItem
from app.langgraph.schemas.structured_extraction import (
    HeaderVerificationResult,
    ItemMatchSelection,
    ItemFieldVerdict,
    ItemVerificationEntry,
    ItemVerificationResult,
)
from app.langgraph.state import GraphState
from app.models.embedding import embed_text_queries
from app.services import pacd_milvus_service, postgres_service

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pass 1: Header Verification
# ──────────────────────────────────────────────────────────────────────────────

_HEADER_SYSTEM_PROMPT = """\
You are a cross-document verification expert comparing a Certificate of Origin (COO) \
against Pre-Arrival Customs Documents (PACD).

You are given:
1. COO header fields (from the certificate being verified)
2. PACD document headers (from previously submitted trade documents)

For EACH COO header field, determine if its value is supported by the PACD evidence.

Rules:
- "match": The COO value matches a corresponding PACD value (case-insensitive, ignore minor whitespace/formatting).
- "mismatch": A related field exists in PACD but the value differs materially.
- "not_found_in_pacd": No corresponding evidence found in any PACD document.
- Compare semantically equivalent fields (e.g. "shipper" = "exporter", "buyer" = "consignee").
- Every COO header field must appear in the output exactly once.
"""


def _pass1_header_verification(
    coo_headers: dict[str, str],
    transaction_id: str,
) -> list[dict[str, Any]]:
    """Pass 1: Verify COO header fields against all PACD document headers."""
    if not coo_headers:
        return []

    # Fetch all PACD document headers for this transaction (SQL — deterministic)
    pacd_docs = postgres_service.get_pacd_documents_by_transaction(transaction_id)

    if not pacd_docs:
        # No PACD documents — everything is not_found
        return [
            {
                "field_key": k,
                "coo_value": v,
                "pacd_value": None,
                "verdict": "not_found_in_pacd",
                "pacd_source_doc": None,
                "pacd_source_filename": None,
            }
            for k, v in coo_headers.items()
        ]

    # Build compact PACD context (~50-100 tokens per document)
    pacd_context_lines = []
    for doc in pacd_docs:
        headers = doc.get("header_fields", {})
        if headers:
            header_str = ", ".join(f"{k}: {v}" for k, v in headers.items())
            pacd_context_lines.append(
                f"[{doc.get('doc_type', 'unknown')} — {doc.get('filename', '?')}]: {header_str}"
            )

    pacd_block = "\n".join(pacd_context_lines)
    coo_block = "\n".join(f"{k}: {v}" for k, v in coo_headers.items())

    user_msg = (
        f"COO Header Fields:\n{coo_block}\n\n"
        f"PACD Document Headers:\n{pacd_block}"
    )

    # Single focused LLM call with structured output
    llm = get_structured_llm(HeaderVerificationResult, temperature=0.0)
    try:
        result: HeaderVerificationResult = llm.invoke([  # type: ignore[assignment]
            SystemMessage(content=_HEADER_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        return [v.model_dump() for v in result.verdicts]
    except Exception as exc:
        logger.warning("cross_reference_node pass1: LLM failed — %s, using fallback", exc)
        # Fallback: simple string matching
        return _fallback_header_match(coo_headers, pacd_docs)


def _fallback_header_match(
    coo_headers: dict[str, str],
    pacd_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Simple string-match fallback for header verification."""
    results = []
    for field_key, coo_value in coo_headers.items():
        found = False
        for doc in pacd_docs:
            headers = doc.get("header_fields", {})
            for pacd_key, pacd_value in headers.items():
                if _normalize(pacd_key) == _normalize(field_key):
                    verdict = "match" if _normalize(coo_value) == _normalize(pacd_value) else "mismatch"
                    results.append({
                        "field_key": field_key,
                        "coo_value": coo_value,
                        "pacd_value": pacd_value,
                        "verdict": verdict,
                        "pacd_source_doc": doc.get("doc_type"),
                        "pacd_source_filename": doc.get("filename"),
                    })
                    found = True
                    break
            if found:
                break
        if not found:
            results.append({
                "field_key": field_key,
                "coo_value": coo_value,
                "pacd_value": None,
                "verdict": "not_found_in_pacd",
                "pacd_source_doc": None,
                "pacd_source_filename": None,
            })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2: Line Item Matching
# ──────────────────────────────────────────────────────────────────────────────

_ITEM_MATCH_SYSTEM_PROMPT = """\
You are selecting the best matching PACD line item for a COO product entry.

Given a COO item and multiple PACD candidate items, select the ONE that best \
matches the COO item based on HS code similarity and product description.

If none of the candidates match the COO item at all, set selected_pacd_item_id to null.
"""


def _pass2_match_items(
    coo_items: list[dict[str, Any]],
    transaction_id: str,
) -> dict[int, dict[str, Any] | None]:
    """Pass 2: Match each COO line item to a PACD line item.

    Strategy:
    1. Exact HS code match via SQL (deterministic, fast)
    2. Broader HS prefix match (first 4 digits) if exact fails
    3. Milvus semantic search on description as final fallback
    4. LLM disambiguation only when multiple candidates exist

    Returns: {coo_item_index: matched_pacd_item_dict_or_None}
    """
    matches: dict[int, dict[str, Any] | None] = {}

    for idx, coo_item in enumerate(coo_items):
        hs_code = coo_item.get("hs_code", "") or ""
        description = coo_item.get("description", "") or ""

        matched_item: dict[str, Any] | None = None

        # Step A: Exact HS code match (SQL)
        if hs_code:
            candidates = postgres_service.get_pacd_line_items_by_transaction(
                transaction_id, hs_code_prefix=hs_code
            )
            if candidates:
                if len(candidates) == 1:
                    matched_item = candidates[0]
                else:
                    # Multiple candidates — use LLM to disambiguate
                    matched_item = _disambiguate_items(coo_item, candidates)

        # Step B: Broader HS prefix (first 4 digits)
        if matched_item is None and hs_code and len(hs_code.replace(".", "")) >= 4:
            prefix_4 = hs_code.replace(".", "").replace(" ", "")[:4]
            candidates = postgres_service.get_pacd_line_items_by_transaction(
                transaction_id, hs_code_prefix=prefix_4
            )
            if candidates:
                if len(candidates) == 1:
                    matched_item = candidates[0]
                else:
                    matched_item = _disambiguate_items(coo_item, candidates)

        # Step C: Milvus semantic search on description (fuzzy fallback)
        if matched_item is None and description:
            query_text = f"{hs_code} | {description}".strip(" |")
            try:
                dense_vecs, sparse_vecs = embed_text_queries([query_text])
                hits = pacd_milvus_service.search_similar_items(
                    transaction_id=transaction_id,
                    dense_vec=dense_vecs[0],
                    sparse_vec=sparse_vecs[0],
                    top_k=10,
                )
                if hits:
                    # Load full item data from Postgres
                    hit_ids = [h["id"] for h in hits]
                    full_items = postgres_service.get_pacd_line_items_by_ids(hit_ids)
                    if len(full_items) == 1:
                        matched_item = full_items[0]
                    elif full_items:
                        matched_item = _disambiguate_items(coo_item, full_items)
            except Exception as exc:
                logger.warning("cross_reference_node pass2: Milvus search failed for item %d: %s", idx, exc)

        matches[idx] = matched_item

    return matches


def _disambiguate_items(
    coo_item: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Use LLM to pick the best match from multiple candidates (~100 tokens)."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Build minimal context for disambiguation
    coo_desc = f"HS: {coo_item.get('hs_code', '?')}, Description: {coo_item.get('description', '?')}"
    candidate_lines = []
    for c in candidates[:5]:  # Cap at 5 to keep context small
        candidate_lines.append(
            f"ID: {c['id']}, HS: {c.get('hs_code', '?')}, Desc: {c.get('description', '?')}"
        )
    candidates_block = "\n".join(candidate_lines)

    user_msg = f"COO Item: {coo_desc}\n\nPACD Candidates:\n{candidates_block}"

    llm = get_structured_llm(ItemMatchSelection, temperature=0.0)
    try:
        result: ItemMatchSelection = llm.invoke([  # type: ignore[assignment]
            SystemMessage(content=_ITEM_MATCH_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        if result.selected_pacd_item_id:
            for c in candidates:
                if str(c["id"]) == result.selected_pacd_item_id:
                    return c
    except Exception as exc:
        logger.warning("cross_reference_node: item disambiguation failed: %s", exc)

    # Fallback: pick highest similarity or first candidate
    return candidates[0]


# ──────────────────────────────────────────────────────────────────────────────
# Pass 3: Per-Item Field Verification
# ──────────────────────────────────────────────────────────────────────────────

_ITEM_VERIFY_SYSTEM_PROMPT = """\
You are a cross-document verification expert.

Compare COO (Certificate of Origin) line items against their matched PACD \
(Pre-Arrival Customs Document) line items.

For each COO item, compare its fields against the matched PACD item's fields.

Rules:
- "match": Values agree (case-insensitive, minor formatting differences OK, \
  e.g. "500 PCS" matches "500 Pieces").
- "mismatch": Values differ materially (different quantity, different HS code digits, etc).
- "not_found_in_pacd": The PACD item doesn't have this field or no PACD match exists.
- Only compare fields that exist in the COO item.
- HS codes: match if the shorter code is a prefix of the longer one \
  (e.g. "8471.30" matches "8471.30.00").
"""


def _pass3_verify_items(
    coo_items: list[dict[str, Any]],
    item_matches: dict[int, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Pass 3: Verify fields of matched item pairs in micro-batches."""
    all_results: list[dict[str, Any]] = []

    # Batch items (max 5 per LLM call to keep context small)
    batch_size = 5
    indices = list(range(len(coo_items)))

    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        batch_context_parts = []

        for idx in batch_indices:
            coo_item = coo_items[idx]
            pacd_item = item_matches.get(idx)

            coo_part = (
                f"COO Item #{coo_item.get('item_number', idx + 1)}:\n"
                f"  HS Code: {coo_item.get('hs_code', 'N/A')}\n"
                f"  Description: {coo_item.get('description', 'N/A')}\n"
                f"  Quantity: {coo_item.get('quantity', 'N/A')}\n"
                f"  Value: {coo_item.get('value', coo_item.get('total_value', 'N/A'))}\n"
                f"  Weight: {coo_item.get('weight', 'N/A')}\n"
                f"  Origin: {coo_item.get('origin_country', 'N/A')}"
            )

            if pacd_item:
                pacd_part = (
                    f"Matched PACD Item (from {pacd_item.get('source_filename', '?')}, "
                    f"line {pacd_item.get('line_number', '?')}):\n"
                    f"  HS Code: {pacd_item.get('hs_code', 'N/A')}\n"
                    f"  Description: {pacd_item.get('description', 'N/A')}\n"
                    f"  Quantity: {pacd_item.get('quantity', 'N/A')} {pacd_item.get('unit', '')}\n"
                    f"  Value: {pacd_item.get('total_value', 'N/A')}\n"
                    f"  Weight: {pacd_item.get('weight', 'N/A')}\n"
                    f"  Origin: {pacd_item.get('origin_country', 'N/A')}"
                )
            else:
                pacd_part = "Matched PACD Item: NONE (no match found)"

            batch_context_parts.append(f"{coo_part}\n{pacd_part}")

        user_msg = "\n\n---\n\n".join(batch_context_parts)

        llm = get_structured_llm(ItemVerificationResult, temperature=0.0)
        try:
            result: ItemVerificationResult = llm.invoke([  # type: ignore[assignment]
                SystemMessage(content=_ITEM_VERIFY_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            for entry in result.items:
                all_results.append(entry.model_dump())
        except Exception as exc:
            logger.warning("cross_reference_node pass3: LLM failed for batch: %s", exc)
            # Fallback: mark all items in this batch as not verifiable
            for idx in batch_indices:
                coo_item = coo_items[idx]
                pacd_item = item_matches.get(idx)
                all_results.append(_fallback_item_verify(coo_item, pacd_item, idx))

    return all_results


def _fallback_item_verify(
    coo_item: dict[str, Any],
    pacd_item: dict[str, Any] | None,
    idx: int,
) -> dict[str, Any]:
    """String-match fallback for a single item comparison."""
    field_verdicts = []
    compare_fields = ["hs_code", "description", "quantity", "weight"]

    for field in compare_fields:
        coo_val = coo_item.get(field)
        if not coo_val:
            continue
        if pacd_item is None:
            field_verdicts.append({
                "field_name": field,
                "coo_value": str(coo_val),
                "pacd_value": None,
                "verdict": "not_found_in_pacd",
            })
        else:
            pacd_val = pacd_item.get(field)
            if pacd_val is None:
                verdict = "not_found_in_pacd"
            elif _normalize(str(coo_val)) == _normalize(str(pacd_val)):
                verdict = "match"
            else:
                verdict = "mismatch"
            field_verdicts.append({
                "field_name": field,
                "coo_value": str(coo_val),
                "pacd_value": str(pacd_val) if pacd_val else None,
                "verdict": verdict,
            })

    return {
        "coo_item_number": coo_item.get("item_number", idx + 1),
        "coo_description": coo_item.get("description", ""),
        "matched_pacd_item_id": str(pacd_item["id"]) if pacd_item else None,
        "matched_pacd_source": pacd_item.get("source_filename") if pacd_item else None,
        "field_verdicts": field_verdicts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(value: str) -> str:
    """Normalize a string for comparison."""
    return str(value).strip().lower().replace("  ", " ")


# ──────────────────────────────────────────────────────────────────────────────
# Main Node
# ──────────────────────────────────────────────────────────────────────────────

def cross_reference_node(state: GraphState) -> GraphState:
    """LangGraph node: 3-pass cross-reference verification.

    Reads
    -----
    state["coo_extracted_data"]    — {"header_fields": {...}, "line_items": [...]}
    state["coo_extracted_fields"]  — flat dict (fallback if structured not available)
    state["transaction_id"]
    state["user_id"]

    Writes
    ------
    state["cross_reference_results"]  — list of DiscrepancyItem dicts
    state["cross_reference_details"]  — full structured results (headers + items)
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("cross_reference_node: starting (v2 — 3-pass structured verification)")

    errors: list[str] = list(state.get("errors") or [])
    transaction_id = state.get("transaction_id") or ""
    user_id = state.get("user_id") or ""

    # Get structured COO data
    coo_data = state.get("coo_extracted_data") or {}
    coo_headers = coo_data.get("header_fields", {})
    coo_items = coo_data.get("line_items", [])

    # Fallback: if no structured data, use flat fields as headers
    if not coo_headers and not coo_items:
        flat_fields = state.get("coo_extracted_fields") or {}
        if flat_fields:
            coo_headers = flat_fields
            logger.info("cross_reference_node: using flat coo_extracted_fields as headers (no structured data)")

    if not coo_headers and not coo_items:
        errors.append("cross_reference_node: no COO data to cross-reference")
        return {**state, "errors": errors, "current_step": "cross_reference_node"}  # type: ignore[return-value]

    # ── Pass 1: Header Verification ──────────────────────────────────────────
    logger.info("cross_reference_node: Pass 1 — header verification (%d fields)", len(coo_headers))
    header_verdicts: list[dict[str, Any]] = []
    if coo_headers:
        try:
            header_verdicts = _pass1_header_verification(coo_headers, transaction_id)
        except Exception as exc:
            errors.append(f"cross_reference_node pass1 failed: {exc}")
            logger.warning("cross_reference_node: pass1 error: %s", exc)

    # ── Pass 2: Item Matching ────────────────────────────────────────────────
    item_verdicts: list[dict[str, Any]] = []
    item_matches: dict[int, dict[str, Any] | None] = {}

    if coo_items:
        logger.info("cross_reference_node: Pass 2 — item matching (%d items)", len(coo_items))
        try:
            item_matches = _pass2_match_items(coo_items, transaction_id)
        except Exception as exc:
            errors.append(f"cross_reference_node pass2 failed: {exc}")
            logger.warning("cross_reference_node: pass2 error: %s", exc)

        # ── Pass 3: Per-Item Verification ────────────────────────────────────
        logger.info("cross_reference_node: Pass 3 — per-item verification")
        try:
            item_verdicts = _pass3_verify_items(coo_items, item_matches)
        except Exception as exc:
            errors.append(f"cross_reference_node pass3 failed: {exc}")
            logger.warning("cross_reference_node: pass3 error: %s", exc)

    # ── Assemble DiscrepancyItem list for report generation ──────────────────
    discrepancy_table: list[DiscrepancyItem] = []

    # Header verdicts → DiscrepancyItems
    for hv in header_verdicts:
        discrepancy_table.append(DiscrepancyItem(
            field_key=hv.get("field_key", ""),
            coo_value=hv.get("coo_value"),
            pacd_value=hv.get("pacd_value"),
            verdict=hv.get("verdict", "not_found_in_pacd"),  # type: ignore[arg-type]
            pacd_source_doc=hv.get("pacd_source_doc") or hv.get("pacd_source_filename"),
            pacd_source_page=None,
        ))

    # Item verdicts → DiscrepancyItems (one per field per item)
    for iv in item_verdicts:
        item_num = iv.get("coo_item_number", "?")
        item_desc = iv.get("coo_description", "")
        source = iv.get("matched_pacd_source")
        for fv in iv.get("field_verdicts", []):
            discrepancy_table.append(DiscrepancyItem(
                field_key=f"item_{item_num}_{fv.get('field_name', '')}",
                coo_value=fv.get("coo_value"),
                pacd_value=fv.get("pacd_value"),
                verdict=fv.get("verdict", "not_found_in_pacd"),  # type: ignore[arg-type]
                pacd_source_doc=source,
                pacd_source_page=None,
            ))

    logger.info(
        "cross_reference_node: completed — %d header verdicts, %d item verdicts, %d total discrepancy entries",
        len(header_verdicts), len(item_verdicts), len(discrepancy_table),
    )

    # Full structured details for debugging / UI
    cross_reference_details = {
        "header_verdicts": header_verdicts,
        "item_verdicts": item_verdicts,
        "item_matches_summary": {
            idx: {"matched": m is not None, "pacd_id": str(m["id"]) if m else None}
            for idx, m in item_matches.items()
        },
    }

    return {  # type: ignore[return-value]
        **state,
        "cross_reference_results": [d.model_dump() for d in discrepancy_table],
        "cross_reference_details": cross_reference_details,
        "errors": errors,
        "current_step": "cross_reference_node",
    }

