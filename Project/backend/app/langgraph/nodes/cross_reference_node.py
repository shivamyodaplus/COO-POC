"""
CrossReferenceNode — for each field extracted from the COO document, searches
the PACD knowledge base (Milvus ``pacd_coo_documents`` collection) and
determines whether the values agree.

Cross-reference logic per field
---------------------------------
1. Build a query string:  "<field_key> <coo_value>"
2. Run hybrid search in pacd_coo_documents filtered by transaction_id.
3. From the top hit, parse ``extracted_kv`` JSON and look for the same key.
4. Compare values (case-insensitive, stripped).
5. Emit a ``DiscrepancyItem`` with verdict: match / mismatch / not_found_in_pacd.

Scope
-----
``scope = "single_transaction"``  — search within the current transaction only.
``scope = "full_history"``        — search all transactions for this user (future use;
                                    infrastructure is ready, just pass extra_transaction_ids).
"""

from __future__ import annotations

import json
import logging

from app.langgraph.schemas.coo_pacd_schemas import (
    CrossReferenceInput,
    CrossReferenceOutput,
    DiscrepancyItem,
)
from app.langgraph.state import GraphState
from app.models.embedding import embed_image
from app.services import pacd_milvus_service
from app.services import transaction_service

logger = logging.getLogger(__name__)


def _normalise(value: str) -> str:
    return str(value).strip().lower()


def _build_query_vec(query_text: str) -> list[float]:
    """Build a dummy dense vector for the text query.

    Because the dense index is built from image embeddings, we return a
    zero vector so only the BM25 (sparse) channel carries weight for
    text-based cross-reference queries.
    """
    return [0.0] * 2048


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

    # --- Determine scope -------------------------------------------------------
    scope = "single_transaction"
    extra_tx_ids: list[str] = []
    if scope == "full_history" and user_id:
        all_txns = transaction_service.list_user_transactions(user_id)
        extra_tx_ids = [t["id"] for t in all_txns if t["id"] != transaction_id]

    # --- Validate input --------------------------------------------------------
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

    # --- Cross-reference each field -------------------------------------------
    discrepancy_table: list[DiscrepancyItem] = []

    for field_key, coo_value in coo_fields.items():
        query_text = f"{field_key} {coo_value}"
        query_vec = _build_query_vec(query_text)

        try:
            hits = pacd_milvus_service.search_by_transaction(
                transaction_id=transaction_id,
                query_text=query_text,
                dense_vec=query_vec,
                top_k=3,
                extra_transaction_ids=extra_tx_ids or None,
            )
        except Exception as exc:
            errors.append(f"cross_reference_node: Milvus search failed for '{field_key}': {exc}")
            discrepancy_table.append(DiscrepancyItem(
                field_key=field_key,
                coo_value=coo_value,
                pacd_value=None,
                verdict="not_found_in_pacd",
            ))
            continue

        if not hits:
            discrepancy_table.append(DiscrepancyItem(
                field_key=field_key,
                coo_value=coo_value,
                pacd_value=None,
                verdict="not_found_in_pacd",
            ))
            continue

        # Scan hits for the matching key in extracted_kv
        pacd_value: str | None = None
        source_doc: str | None = None
        source_page: int | None = None

        for hit in hits:
            kv_raw = hit.get("extracted_kv", "[]")
            try:
                kv_list = json.loads(kv_raw) if isinstance(kv_raw, str) else kv_raw
            except json.JSONDecodeError:
                continue
            for kv in kv_list:
                if _normalise(kv.get("key", "")) == _normalise(field_key):
                    pacd_value = str(kv.get("value", ""))
                    source_doc = hit.get("filename")
                    source_page = hit.get("page_num")
                    break
            if pacd_value is not None:
                break

        if pacd_value is None:
            verdict: str = "not_found_in_pacd"
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
        "errors": output.errors,
        "current_step": "cross_reference_node",
    }
