"""
RAGRetrievalNode — Reduce phase, step 2 of the Map-Reduce RAG pipeline.

Executes targeted multi-category hybrid search against the ``pacd_document_chunks``
Milvus collection and applies table isolation to ensure full table context.

Algorithm
---------
1. Run ``pacd_milvus_service.search_document_chunks_multi()`` with the four
   query categories produced by ``multi_query_expansion_node``.  Each category
   maps to a specific ``chunk_type`` filter, so irrelevant sections are never
   retrieved:
     - item_queries  → chunk_type in ["table", "header"]
     - header_queries → chunk_type == "header"
     - footer_queries → chunk_type == "footer"
     - hs_queries     → chunk_type == "table"

2. **Table isolation**: collect the unique ``document_id`` values of every
   table chunk in the initial result set, then fetch all sibling table chunks
   from those same logical documents.  This prevents the scenario where a
   20-item invoice is split into 3 table chunks and only the first chunk is
   returned, leaving the LLM with an incomplete view of the table.

3. Merge the initial results and the sibling table chunks, re-rank by RRF
   score (siblings that were not in the initial set receive a small additive
   bonus so they appear near their retrieved sibling), and cap at
   ``RAG_FINAL_TOP_K * TABLE_ISOLATION_MAX_CHUNKS / MAX_TABLE_ITEMS_PER_CHUNK``
   to prevent context explosion.

4. The final list is de-duplicated and stored in ``state["retrieved_chunks"]``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.langgraph.state import GraphState
from app.services import pacd_milvus_service

logger = logging.getLogger(__name__)


def rag_retrieval_node(state: GraphState) -> GraphState:
    """LangGraph node: multi-query hybrid search + table isolation.

    Reads
    -----
    state["multi_queries"]   — {"item": [...], "header": [...], "footer": [...], "hs": [...]}
    state["transaction_id"]

    Writes
    ------
    state["retrieved_chunks"]  — ranked list of chunk dicts with rrf_score
    state["current_step"]
    Appends to state["errors"]
    """
    logger.info("rag_retrieval_node: starting")

    errors: list[str] = list(state.get("errors") or [])
    multi_queries: dict[str, list[str]] = state.get("multi_queries") or {}
    transaction_id = state.get("transaction_id") or ""

    if not transaction_id:
        errors.append("rag_retrieval_node: transaction_id is missing — cannot search Milvus")
        return {
            **state,
            "retrieved_chunks": [],
            "errors": errors,
            "current_step": "rag_retrieval_node",
        }  # type: ignore[return-value]

    if not multi_queries:
        errors.append("rag_retrieval_node: multi_queries is empty — skipping retrieval")
        return {
            **state,
            "retrieved_chunks": [],
            "errors": errors,
            "current_step": "rag_retrieval_node",
        }  # type: ignore[return-value]

    # ── Step 1: Multi-category hybrid search ─────────────────────────────
    initial_results: list[dict[str, Any]] = []
    try:
        initial_results = pacd_milvus_service.search_document_chunks_multi(
            transaction_id=transaction_id,
            queries_by_type=multi_queries,
            top_k=settings.RAG_RETRIEVAL_TOP_K,
            top_t=settings.RAG_FINAL_TOP_K,
        )
        logger.info(
            "rag_retrieval_node: initial retrieval returned %d chunks", len(initial_results)
        )
    except Exception as exc:
        errors.append(f"rag_retrieval_node: Milvus search failed — {exc}")
        logger.error("rag_retrieval_node: search error: %s", exc)
        return {
            **state,
            "retrieved_chunks": [],
            "errors": errors,
            "current_step": "rag_retrieval_node",
        }  # type: ignore[return-value]

    # ── Step 2: Table isolation ───────────────────────────────────────────
    # Find document_ids of retrieved table chunks, then fetch all siblings.
    table_doc_ids: list[str] = list({
        c["document_id"]
        for c in initial_results
        if c.get("chunk_type") == "table" and c.get("document_id")
    })

    existing_ids: set[str] = {c["id"] for c in initial_results if c.get("id")}
    siblings: list[dict[str, Any]] = []

    if table_doc_ids:
        try:
            all_siblings = pacd_milvus_service.fetch_sibling_table_chunks(
                transaction_id=transaction_id,
                document_ids=table_doc_ids,
                max_per_doc=settings.TABLE_ISOLATION_MAX_CHUNKS,
            )
            # Include only chunks that are NOT already in the initial result set
            siblings = [s for s in all_siblings if s.get("id") not in existing_ids]
            logger.info(
                "rag_retrieval_node: table isolation fetched %d siblings (%d new) from %d docs",
                len(all_siblings), len(siblings), len(table_doc_ids),
            )
        except Exception as exc:
            errors.append(f"rag_retrieval_node: table isolation failed — {exc}")
            logger.warning("rag_retrieval_node: table isolation error: %s", exc)

    # ── Step 3: Merge, assign scores, de-duplicate ────────────────────────
    # Siblings get a small default RRF score so they appear just after the
    # retrieved chunk from the same document.
    min_rrf = min((c.get("rrf_score", 0.0) for c in initial_results), default=0.0)
    sibling_score = max(0.001, min_rrf * 0.5)

    merged: list[dict[str, Any]] = list(initial_results)
    for sib in siblings:
        sib["rrf_score"] = sibling_score
        merged.append(sib)

    # Sort by (document_id, chunk_index) within same doc, else by rrf_score desc
    # Strategy: first sort by doc+index (for readability), then stable-sort by score
    merged.sort(key=lambda c: (c.get("document_id", ""), c.get("chunk_index", 0)))
    merged.sort(key=lambda c: c.get("rrf_score", 0.0), reverse=True)

    # Hard cap to prevent runaway context
    max_chunks = settings.RAG_FINAL_TOP_K + settings.TABLE_ISOLATION_MAX_CHUNKS
    retrieved = merged[:max_chunks]

    logger.info(
        "rag_retrieval_node: final retrieved_chunks = %d (cap=%d)",
        len(retrieved), max_chunks,
    )

    return {
        **state,
        "retrieved_chunks": retrieved,
        "errors": errors,
        "current_step": "rag_retrieval_node",
    }  # type: ignore[return-value]
