"""
Parallel Search Node — Phase 2b of 2-Tier GraphRAG.

Embeds validation queries with BGE-M3 and runs parallel hybrid search
against the ``coo_reference_chunks`` collection, filtering by transaction_id.
Deduplicates results by chunk_id (keeps highest score).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.core.config import settings
from app.langgraph.schemas.graphrag_schemas import CONTENT_TYPE_ORDER
from app.langgraph.state import GraphState
from app.models.embedding import embed_text_queries
from app.services.pacd_milvus_service import hybrid_search_reference

logger = logging.getLogger(__name__)


def _search_single_query(
    query_text: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    transaction_id: str,
) -> list[dict[str, Any]]:
    """Run hybrid search for a single query."""
    return hybrid_search_reference(
        transaction_id=transaction_id,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        top_k=settings.RAG_RETRIEVAL_TOP_K,
        threshold=settings.SCORE_THRESHOLD,
    )


def parallel_search_node(state: GraphState) -> dict[str, Any]:
    """Run parallel hybrid search for all validation queries."""
    validation_queries: list[str] = state.get("validation_queries") or []
    transaction_id: str = state.get("transaction_id") or ""

    if not validation_queries:
        logger.warning("parallel_search_node: no validation queries")
        return {
            "retrieved_chunks": [],
            "errors": state.get("errors", []) + ["No validation queries to search"],
            "current_step": "parallel_search_node",
        }

    if not transaction_id:
        logger.warning("parallel_search_node: no transaction_id")
        return {
            "retrieved_chunks": [],
            "errors": state.get("errors", []) + ["No transaction_id for search"],
            "current_step": "parallel_search_node",
        }

    # Batch-embed all queries at once
    try:
        dense_vecs, sparse_vecs = embed_text_queries(validation_queries)
    except Exception as exc:
        logger.error("parallel_search_node: embedding failed — %s", exc)
        return {
            "retrieved_chunks": [],
            "errors": state.get("errors", []) + [f"Query embedding failed: {exc}"],
            "current_step": "parallel_search_node",
        }

    # Parallel hybrid search
    all_hits: list[dict[str, Any]] = []
    max_workers = min(len(validation_queries), 8)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_query = {}
        for i, query in enumerate(validation_queries):
            future = pool.submit(
                _search_single_query,
                query,
                dense_vecs[i],
                sparse_vecs[i],
                transaction_id,
            )
            future_to_query[future] = query

        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                hits = future.result()
                logger.debug(
                    "parallel_search_node: Q='%s' → %d hits", query[:50], len(hits)
                )
                all_hits.extend(hits)
            except Exception as exc:
                logger.warning(
                    "parallel_search_node: search failed for '%s' — %s", query[:50], exc
                )

    # Deduplicate by chunk_id — keep highest-scoring hit
    best: dict[str, dict[str, Any]] = {}
    for hit in all_hits:
        cid = hit.get("chunk_id")
        if not cid:
            continue
        if cid not in best or hit["score"] > best[cid]["score"]:
            best[cid] = hit

    # Sort by source_document → page_number → content_type (H→C→F) → chunk_index
    # This ensures the validation LLM sees data grouped by file, in reading order
    def _sort_key(chunk: dict[str, Any]) -> tuple[str, int, int, int]:
        meta = {}
        if chunk.get("metadata_json"):
            try:
                meta = json.loads(chunk["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        source = meta.get("source_document", "")
        page = meta.get("page_number", 999)
        ctype = CONTENT_TYPE_ORDER.get(meta.get("content_type", ""), 9)
        cidx = meta.get("chunk_index", 0)
        return (source, page, ctype, cidx)

    retrieved_chunks = sorted(best.values(), key=_sort_key)

    logger.info(
        "parallel_search_node: %d queries → %d total hits → %d unique chunks",
        len(validation_queries), len(all_hits), len(retrieved_chunks),
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "errors": state.get("errors", []),
        "current_step": "parallel_search_node",
    }
