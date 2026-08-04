"""
CrossReferenceNode — validates COO fields against the PACD knowledge base.

Algorithm
---------
1. For each COO field, embed  ``"{field}: {value}"`` with BGE-M3 (dense+sparse).
2. Run all field searches concurrently (ThreadPoolExecutor), top_k=2 per field.
3. Collect all hits, deduplicate by chunk_id (keep highest-score copy).
4. Assemble a context block from the unique chunks.
5. Send the context + COO fields to the text LLM for structured validation.
6. Parse the LLM response into DiscrepancyItem list.

Fallback: if the LLM call fails, falls back to the simple string-comparison
approach so the pipeline never blocks on an LLM outage.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.langgraph.llm import get_llm
from app.langgraph.schemas.coo_pacd_schemas import (
    CrossReferenceInput,
    CrossReferenceOutput,
    DiscrepancyItem,
)
from app.langgraph.state import GraphState
from app.models.embedding import embed_text_queries
from app.services import pacd_milvus_service
from app.services import transaction_service

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a cross-document verification expert.
Given PACD (Pre-Approved Commercial Document) evidence chunks and COO (Certificate of Origin) \
field values, determine for each COO field whether its value is supported, contradicted, or \
absent in the PACD evidence.

Return ONLY a JSON array — no markdown, no prose, no extra keys.

Each element:
{
  "field_key": "<exact COO field name>",
  "coo_value": "<value from COO>",
  "pacd_value": "<matching value found in PACD, or null>",
  "verdict": "<match|mismatch|not_found_in_pacd>",
  "pacd_source_doc": "<filename from the matching chunk, or null>",
  "pacd_source_page": <page number integer, or null>
}

Rules:
- "match": COO value matches PACD value (case-insensitive, ignore extra whitespace).
- "mismatch": a related value was found in PACD but differs from the COO value.
- "not_found_in_pacd": no relevant evidence in the provided chunks.
- Every COO field must appear in the output exactly once.
"""


def _normalise(value: str) -> str:
    return str(value).strip().lower()


def _search_one_field(
    field_key: str,
    coo_value: str,
    transaction_id: str,
    extra_tx_ids: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    """Embed one field and run Milvus search. Returns (field_key, hits)."""
    query_text = f"{field_key}: {coo_value}"
    try:
        dense_vecs, sparse_vecs = embed_text_queries([query_text])
        hits = pacd_milvus_service.search_by_transaction(
            transaction_id=transaction_id,
            dense_vec=dense_vecs[0],
            sparse_vec=sparse_vecs[0],
            top_k=2,
            extra_transaction_ids=extra_tx_ids or None,
        )
    except Exception as exc:
        logger.warning("cross_reference_node: search failed for '%s': %s", field_key, exc)
        hits = []
    return field_key, hits


def _llm_validate(
    coo_fields: dict[str, str],
    unique_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Call the text LLM to produce a structured verdict for all fields."""
    # Build context block
    context_lines = []
    for chunk in unique_chunks:
        src = f"{chunk.get('filename', 'unknown')}, page {chunk.get('page_num', '?')}"
        context_lines.append(f"[{src}]\n{chunk.get('chunk_text', '')}")
    context_block = "\n\n---\n\n".join(context_lines) or "(no PACD evidence found)"

    # Build COO fields block
    fields_block = "\n".join(f"{k}: {v}" for k, v in coo_fields.items())

    user_msg = (
        f"PACD Evidence Chunks:\n{context_block}\n\n"
        f"COO Document Fields:\n{fields_block}"
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    raw = response.content if hasattr(response, "content") else str(response)

    # Strip optional markdown fencing
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    return json.loads(raw)


def cross_reference_node(state: GraphState) -> GraphState:
    """LangGraph node: cross-reference COO fields against PACD knowledge base.

    Reads
    -----
    state["coo_extracted_fields"]  — {field_key: value}
    state["transaction_id"], state["user_id"]
    state["errors"]

    Writes
    ------
    state["cross_reference_results"] — list of DiscrepancyItem dicts
    state["cross_reference_chunks"]  — deduplicated PACD chunks used in validation
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("cross_reference_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    coo_fields: dict[str, str] = state.get("coo_extracted_fields") or {}
    transaction_id = state.get("transaction_id") or ""
    user_id = state.get("user_id") or ""

    if not coo_fields:
        errors.append("cross_reference_node: no COO fields to cross-reference")
        return {**state, "errors": errors, "current_step": "cross_reference_node"}  # type: ignore[return-value]

    # --- Scope -------------------------------------------------------------------
    scope = "single_transaction"
    extra_tx_ids: list[str] = []
    if scope == "full_history" and user_id:
        all_txns = transaction_service.list_user_transactions(user_id)
        extra_tx_ids = [t["id"] for t in all_txns if t["id"] != transaction_id]

    try:
        CrossReferenceInput(
            transaction_id=transaction_id,
            user_id=user_id,
            coo_extracted_fields=coo_fields,
            scope=scope,  # type: ignore[arg-type]
        )
    except Exception as exc:
        errors.append(f"cross_reference_node validation error: {exc}")
        return {**state, "errors": errors, "current_step": "cross_reference_node"}  # type: ignore[return-value]

    # --- Concurrent search: top-2 per field -------------------------------------
    max_workers = min(len(coo_fields), 8)
    all_hits_by_field: dict[str, list[dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _search_one_field, fk, fv, transaction_id, extra_tx_ids
            ): fk
            for fk, fv in coo_fields.items()
        }
        for future in as_completed(futures):
            field_key, hits = future.result()
            all_hits_by_field[field_key] = hits

    # --- Deduplicate by chunk_id ------------------------------------------------
    seen_ids: dict[str, dict[str, Any]] = {}
    for hits in all_hits_by_field.values():
        for hit in hits:
            chunk_id = hit.get("id") or ""
            if not chunk_id:
                continue
            existing = seen_ids.get(chunk_id)
            if existing is None or hit.get("similarity", 0) > existing.get("similarity", 0):
                seen_ids[chunk_id] = hit

    unique_chunks = list(seen_ids.values())
    logger.info(
        "cross_reference_node: %d unique chunks from %d fields",
        len(unique_chunks), len(coo_fields),
    )

    # --- LLM validation ---------------------------------------------------------
    discrepancy_table: list[DiscrepancyItem] = []

    try:
        llm_results = _llm_validate(coo_fields, unique_chunks)
        for item in llm_results:
            discrepancy_table.append(DiscrepancyItem(
                field_key=item.get("field_key", ""),
                coo_value=item.get("coo_value", ""),
                pacd_value=item.get("pacd_value"),
                verdict=item.get("verdict", "not_found_in_pacd"),  # type: ignore[arg-type]
                pacd_source_doc=item.get("pacd_source_doc"),
                pacd_source_page=item.get("pacd_source_page"),
            ))
    except Exception as exc:
        errors.append(f"cross_reference_node: LLM validation failed ({exc}) — falling back to string match")
        logger.warning("cross_reference_node: LLM fallback: %s", exc)
        # Fallback: simple string comparison using retrieved chunks
        for field_key, coo_value in coo_fields.items():
            hits = all_hits_by_field.get(field_key, [])
            pacd_value: str | None = None
            source_doc: str | None = None
            source_page: int | None = None

            for hit in hits:
                chunk_text = hit.get("chunk_text", "")
                # Try to extract value from "key: value" format in chunk
                for line in chunk_text.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        if _normalise(k) == _normalise(field_key):
                            pacd_value = v.strip()
                            source_doc = hit.get("filename")
                            source_page = hit.get("page_num")
                            break
                if pacd_value is not None:
                    break

            if pacd_value is None:
                verdict = "not_found_in_pacd"
            elif _normalise(coo_value) == _normalise(pacd_value):
                verdict = "match"
            else:
                verdict = "mismatch"

            discrepancy_table.append(DiscrepancyItem(
                field_key=field_key,
                coo_value=coo_value,
                pacd_value=pacd_value,
                verdict=verdict,  # type: ignore[arg-type]
                pacd_source_doc=source_doc,
                pacd_source_page=source_page,
            ))

    matched = sum(1 for d in discrepancy_table if d.verdict == "match")
    mismatched = sum(1 for d in discrepancy_table if d.verdict == "mismatch")
    not_found = sum(1 for d in discrepancy_table if d.verdict == "not_found_in_pacd")

    output = CrossReferenceOutput(
        discrepancy_table=discrepancy_table,
        total_fields=len(discrepancy_table),
        matched=matched,
        mismatched=mismatched,
        not_found=not_found,
        errors=errors,
    )

    logger.info(
        "cross_reference_node: total=%d match=%d mismatch=%d not_found=%d",
        output.total_fields, output.matched, output.mismatched, output.not_found,
    )

    return {  # type: ignore[return-value]
        **state,
        "cross_reference_results": [d.model_dump() for d in output.discrepancy_table],
        "cross_reference_chunks": unique_chunks,
        "errors": output.errors,
        "current_step": "cross_reference_node",
    }

