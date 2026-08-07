"""
Milvus service for PACD document chunk search.

Two collections are managed here:

1. ``pacd_coo_documents`` (legacy fuzzy fallback — kept for backward compat)
   - Line-item description embeddings for HS-code fallback search.

2. ``pacd_document_chunks`` (Map-Reduce RAG — primary collection)
   - Layout-aware granular chunks: header / table(s) / footer per logical doc.
   - Table chunks are split into groups of MAX_TABLE_ITEMS_PER_CHUNK items so
     large invoices (50+ pages, 500+ items) are indexed without hitting token
     or VARCHAR limits.
   - Schema:
       id               VARCHAR(256)  PK  — "{pg_doc_id}_{chunk_type}_{index}"
       transaction_id   VARCHAR(128)      — INVERTED (primary scope filter)
       file_id          VARCHAR(256)      — INVERTED (SHA-256 of the uploaded file)
       document_id      VARCHAR(128)      — INVERTED (Postgres logical doc UUID, for sibling lookup)
       doc_type         VARCHAR(64)       — "commercial_invoice" etc.
       page_numbers     VARCHAR(256)      — JSON: "[4, 5, 6]"
       chunk_type       VARCHAR(16)       — "table" | "header" | "footer"  INVERTED
       chunk_index      INT64             — ordering within logical document
       chunk_text       VARCHAR(8192)     — markdown table or KV pairs
       chunk_metadata   VARCHAR(4096)     — JSON: {hs_codes, extracted_keys, …}
       dense            FLOAT_VECTOR(1024)
       sparse           SPARSE_FLOAT_VECTOR
"""

from __future__ import annotations

from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    WeightedRanker,
)

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

COLLECTION = settings.PACD_MILVUS_COLLECTION            # pacd_coo_documents (legacy)
CHUNKS_COLLECTION = "pacd_chunks"                        # old section chunks (deprecated)
DOCUMENT_CHUNKS_COLLECTION = settings.PACD_CHUNK_COLLECTION  # pacd_document_chunks (new)
DENSE_DIM = 1024  # BGE-M3

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    if _client is None:
        raise RuntimeError("PACD Milvus client not initialised. Call connect() first.")
    return _client


def connect() -> None:
    global _client
    _client = MilvusClient(uri=settings.MILVUS_URI)
    logger.info("PACD Milvus: connected at %s", settings.MILVUS_URI)


def _has_v3_schema(client: MilvusClient) -> bool:
    """Return True if the collection uses the v3 description-only schema."""
    try:
        desc = client.describe_collection(COLLECTION)
        field_names = {f.get("name", "") for f in desc.get("fields", [])}
        return "description_text" in field_names and "chunk_text" not in field_names
    except Exception:
        return False


def ensure_pacd_collection() -> None:
    """Create or recreate the PACD item description collection (v3 schema)."""
    client = get_client()

    if client.has_collection(COLLECTION):
        if _has_v3_schema(client):
            logger.info("PACD collection '%s' exists with v3 schema — skipping creation", COLLECTION)
            return
        logger.warning(
            "PACD collection '%s' has old schema — dropping and recreating with v3", COLLECTION
        )
        client.drop_collection(COLLECTION)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",               DataType.VARCHAR,      is_primary=True, max_length=128)
    schema.add_field("transaction_id",   DataType.VARCHAR,      max_length=128)
    schema.add_field("hs_code",          DataType.VARCHAR,      max_length=20)
    schema.add_field("description_text", DataType.VARCHAR,      max_length=2048)
    schema.add_field("dense",            DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse",           DataType.SPARSE_FLOAT_VECTOR)

    idx = client.prepare_index_params()
    idx.add_index("dense",          index_type="HNSW",                  metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    idx.add_index("sparse",         index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    idx.add_index("transaction_id", index_type="INVERTED")

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=idx,
    )
    logger.info("PACD Milvus: created collection '%s' (v3 description-only schema)", COLLECTION)


def upsert_item_descriptions(records: list[dict[str, Any]]) -> None:
    """Upsert line item description records.

    Each record must contain: id, transaction_id, hs_code, description_text, dense, sparse.
    """
    if not records:
        return
    get_client().upsert(collection_name=COLLECTION, data=records)


def search_similar_items(
    transaction_id: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Hybrid search for similar item descriptions within a transaction.

    Used as fuzzy fallback when exact HS code SQL lookup fails.
    """
    client = get_client()

    expr = f'transaction_id == "{transaction_id}"'
    out_fields = ["id", "transaction_id", "hs_code", "description_text"]

    if not sparse_vec:
        results = client.search(
            collection_name=COLLECTION,
            data=[dense_vec],
            anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            filter=expr,
            output_fields=out_fields,
        )
    else:
        alpha = settings.HYBRID_ALPHA
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            expr=expr,
        )
        results = client.hybrid_search(
            collection_name=COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(alpha, 1.0 - alpha),
            limit=top_k,
            output_fields=out_fields,
        )

    hits: list[dict[str, Any]] = []
    for hit in (results[0] if results else []):
        if isinstance(hit, dict):
            entity = hit
            distance = hit.get("distance", 0.0)
        else:
            entity = hit.entity if hasattr(hit, "entity") else {}
            distance = hit.distance

        hits.append({
            "id":               entity.get("id"),
            "transaction_id":   entity.get("transaction_id"),
            "hs_code":          entity.get("hs_code", ""),
            "description_text": entity.get("description_text", ""),
            "similarity":       round(float(distance), 6),
        })
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# pacd_chunks collection — semantic section chunks for parallel verification
# ──────────────────────────────────────────────────────────────────────────────

def ensure_pacd_chunks_collection() -> None:
    """Create (or verify) the pacd_chunks collection.

    Schema
    ------
    id             VARCHAR(128)  PK  — "{doc_id}_{chunk_type}"
    transaction_id VARCHAR(128)      — transaction scope filter
    document_id    VARCHAR(128)      — source PACD logical document id
    doc_type       VARCHAR(64)       — invoice / packing_list / etc.
    chunk_type     VARCHAR(16)       — "header" | "table" | "footer"
    chunk_text     VARCHAR(4096)     — serialized text (embedded)
    chunk_metadata VARCHAR(4096)     — JSON string of raw structured data
    dense          FLOAT_VECTOR(1024)
    sparse         SPARSE_FLOAT_VECTOR
    """
    client = get_client()

    if client.has_collection(CHUNKS_COLLECTION):
        logger.info("pacd_chunks collection '%s' already exists — skipping creation", CHUNKS_COLLECTION)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",             DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("transaction_id", DataType.VARCHAR, max_length=128)
    schema.add_field("document_id",    DataType.VARCHAR, max_length=128)
    schema.add_field("doc_type",       DataType.VARCHAR, max_length=64)
    schema.add_field("chunk_type",     DataType.VARCHAR, max_length=16)
    schema.add_field("chunk_text",     DataType.VARCHAR, max_length=4096)
    schema.add_field("chunk_metadata", DataType.VARCHAR, max_length=4096)
    schema.add_field("dense",          DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse",         DataType.SPARSE_FLOAT_VECTOR)

    idx = client.prepare_index_params()
    idx.add_index("dense",          index_type="HNSW",                  metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    idx.add_index("sparse",         index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    idx.add_index("transaction_id", index_type="INVERTED")
    idx.add_index("chunk_type",     index_type="INVERTED")

    client.create_collection(
        collection_name=CHUNKS_COLLECTION,
        schema=schema,
        index_params=idx,
    )
    logger.info("Created pacd_chunks collection '%s'", CHUNKS_COLLECTION)


def upsert_chunks(records: list[dict[str, Any]]) -> None:
    """Upsert section chunk records into pacd_chunks.

    Each record must have: id, transaction_id, document_id, doc_type,
    chunk_type, chunk_text, chunk_metadata, dense, sparse.
    """
    if not records:
        return
    get_client().upsert(collection_name=CHUNKS_COLLECTION, data=records)
    logger.debug("Upserted %d chunk records into '%s'", len(records), CHUNKS_COLLECTION)


def search_section_chunks(
    section: str,
    queries: list[str],
    transaction_id: str,
    top_k: int = 10,
    top_t: int = 15,
) -> list[dict[str, Any]]:
    """Multi-query hybrid search on a specific section, fused with RRF.

    Parameters
    ----------
    section        : "header" | "table" | "footer"
    queries        : list of natural-language query strings
    transaction_id : filter scope
    top_k          : candidates per individual query
    top_t          : final unique chunks to return after RRF fusion

    Returns a list of chunk dicts sorted by RRF score (descending), each
    containing: id, document_id, doc_type, chunk_type, chunk_text,
    chunk_metadata, rrf_score.
    """
    from app.models.embedding import embed_text_queries  # local import to avoid circular

    if not queries:
        return []

    client = get_client()
    expr = f'transaction_id == "{transaction_id}" && chunk_type == "{section}"'
    out_fields = ["id", "transaction_id", "document_id", "doc_type",
                  "chunk_type", "chunk_text", "chunk_metadata"]

    # Batch-embed all queries at once
    try:
        dense_vecs, sparse_vecs = embed_text_queries(queries)
    except Exception as exc:
        logger.warning("search_section_chunks: embedding failed — %s", exc)
        return []

    all_ranked: list[list[dict[str, Any]]] = []

    for q_idx, (dense_vec, sparse_vec) in enumerate(zip(dense_vecs, sparse_vecs)):
        try:
            if not sparse_vec:
                results = client.search(
                    collection_name=CHUNKS_COLLECTION,
                    data=[dense_vec],
                    anns_field="dense",
                    search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                    limit=top_k,
                    filter=expr,
                    output_fields=out_fields,
                )
                raw_hits = results[0] if results else []
            else:
                alpha = settings.HYBRID_ALPHA
                dense_req = AnnSearchRequest(
                    data=[dense_vec],
                    anns_field="dense",
                    param={"metric_type": "COSINE", "params": {"ef": 100}},
                    limit=top_k,
                    expr=expr,
                )
                sparse_req = AnnSearchRequest(
                    data=[sparse_vec],
                    anns_field="sparse",
                    param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                    limit=top_k,
                    expr=expr,
                )
                results = client.hybrid_search(
                    collection_name=CHUNKS_COLLECTION,
                    reqs=[dense_req, sparse_req],
                    ranker=WeightedRanker(alpha, 1.0 - alpha),
                    limit=top_k,
                    output_fields=out_fields,
                )
                raw_hits = results[0] if results else []

            hits: list[dict[str, Any]] = []
            for hit in raw_hits:
                entity = hit if isinstance(hit, dict) else (hit.entity if hasattr(hit, "entity") else {})
                hits.append({
                    "id":             entity.get("id"),
                    "transaction_id": entity.get("transaction_id"),
                    "document_id":    entity.get("document_id"),
                    "doc_type":       entity.get("doc_type", ""),
                    "chunk_type":     entity.get("chunk_type", section),
                    "chunk_text":     entity.get("chunk_text", ""),
                    "chunk_metadata": entity.get("chunk_metadata", "{}"),
                })
            all_ranked.append(hits)

        except Exception as exc:
            logger.warning("search_section_chunks: query %d failed — %s", q_idx, exc)
            all_ranked.append([])

    # ── Reciprocal Rank Fusion (k=60) ──────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, dict[str, Any]] = {}

    for ranked_list in all_ranked:
        for rank, hit in enumerate(ranked_list, start=1):
            cid = hit.get("id")
            if not cid:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank)
            if cid not in chunk_store:
                chunk_store[cid] = hit

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    result: list[dict[str, Any]] = []
    for cid in sorted_ids[:top_t]:
        chunk = dict(chunk_store[cid])
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        result.append(chunk)

    logger.debug(
        "search_section_chunks: section=%s queries=%d unique_chunks=%d returned=%d",
        section, len(queries), len(rrf_scores), len(result),
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# pacd_document_chunks — Map-Reduce RAG collection (new, replaces pacd_chunks)
# ──────────────────────────────────────────────────────────────────────────────

def _has_document_chunks_schema(client: MilvusClient) -> bool:
    """Return True if the new pacd_document_chunks schema is in place."""
    try:
        desc = client.describe_collection(DOCUMENT_CHUNKS_COLLECTION)
        field_names = {f.get("name", "") for f in desc.get("fields", [])}
        return "file_id" in field_names and "chunk_index" in field_names
    except Exception:
        return False


def ensure_pacd_document_chunks_collection() -> None:
    """Create (or verify) the ``pacd_document_chunks`` collection.

    This is the primary PACD index for the Map-Reduce RAG pipeline.
    Drops and recreates if an incompatible schema is found.
    """
    client = get_client()

    if client.has_collection(DOCUMENT_CHUNKS_COLLECTION):
        if _has_document_chunks_schema(client):
            logger.info(
                "pacd_document_chunks collection '%s' exists — skipping creation",
                DOCUMENT_CHUNKS_COLLECTION,
            )
            return
        logger.warning(
            "pacd_document_chunks collection '%s' has old schema — dropping and recreating",
            DOCUMENT_CHUNKS_COLLECTION,
        )
        client.drop_collection(DOCUMENT_CHUNKS_COLLECTION)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",             DataType.VARCHAR,      is_primary=True, max_length=256)
    schema.add_field("transaction_id", DataType.VARCHAR,      max_length=128)
    schema.add_field("file_id",        DataType.VARCHAR,      max_length=256)
    schema.add_field("document_id",    DataType.VARCHAR,      max_length=128)
    schema.add_field("doc_type",       DataType.VARCHAR,      max_length=64)
    schema.add_field("page_numbers",   DataType.VARCHAR,      max_length=256)
    schema.add_field("chunk_type",     DataType.VARCHAR,      max_length=16)
    schema.add_field("chunk_index",    DataType.INT64)
    schema.add_field("chunk_text",     DataType.VARCHAR,      max_length=8192)
    schema.add_field("chunk_metadata", DataType.VARCHAR,      max_length=4096)
    schema.add_field("dense",          DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse",         DataType.SPARSE_FLOAT_VECTOR)

    idx = client.prepare_index_params()
    idx.add_index("dense",          index_type="HNSW",                  metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    idx.add_index("sparse",         index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    idx.add_index("transaction_id", index_type="INVERTED")
    idx.add_index("file_id",        index_type="INVERTED")
    idx.add_index("document_id",    index_type="INVERTED")
    idx.add_index("chunk_type",     index_type="INVERTED")

    client.create_collection(
        collection_name=DOCUMENT_CHUNKS_COLLECTION,
        schema=schema,
        index_params=idx,
    )
    logger.info("Created pacd_document_chunks collection '%s'", DOCUMENT_CHUNKS_COLLECTION)


def upsert_document_chunks(records: list[dict[str, Any]]) -> None:
    """Upsert layout-aware chunk records into ``pacd_document_chunks``.

    Each record must have: id, transaction_id, file_id, document_id, doc_type,
    page_numbers, chunk_type, chunk_index, chunk_text, chunk_metadata, dense, sparse.
    """
    if not records:
        return
    get_client().upsert(collection_name=DOCUMENT_CHUNKS_COLLECTION, data=records)
    logger.debug("Upserted %d records into '%s'", len(records), DOCUMENT_CHUNKS_COLLECTION)


def fetch_sibling_table_chunks(
    transaction_id: str,
    document_ids: list[str],
    max_per_doc: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch all table chunks belonging to the given logical document IDs.

    Used for table isolation: when a table chunk is retrieved by semantic
    search, pull *all* its siblings so the LLM sees the complete table.

    Parameters
    ----------
    transaction_id : scope filter
    document_ids   : list of Postgres logical document UUIDs
    max_per_doc    : cap on sibling chunks per document (None = unlimited)
    """
    if not document_ids:
        return []

    client = get_client()
    out_fields = [
        "id", "transaction_id", "file_id", "document_id", "doc_type",
        "page_numbers", "chunk_type", "chunk_index", "chunk_text", "chunk_metadata",
    ]

    all_siblings: list[dict[str, Any]] = []
    for doc_id in document_ids:
        expr = (
            f'transaction_id == "{transaction_id}" && '
            f'document_id == "{doc_id}" && '
            f'chunk_type == "table"'
        )
        try:
            rows = client.query(
                collection_name=DOCUMENT_CHUNKS_COLLECTION,
                filter=expr,
                output_fields=out_fields,
                limit=max_per_doc or 1000,
            )
            for row in rows:
                all_siblings.append(dict(row))
        except Exception as exc:
            logger.warning(
                "fetch_sibling_table_chunks: query failed for doc_id=%s — %s", doc_id, exc
            )

    # Sort by (document_id, chunk_index) for deterministic ordering
    all_siblings.sort(key=lambda r: (r.get("document_id", ""), r.get("chunk_index", 0)))
    return all_siblings


def _run_single_hybrid_search(
    client: MilvusClient,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    expr: str,
    top_k: int,
    out_fields: list[str],
) -> list[dict[str, Any]]:
    """Execute one hybrid (or dense-only) search and return normalised hit dicts."""
    alpha = settings.HYBRID_ALPHA
    if not sparse_vec:
        results = client.search(
            collection_name=DOCUMENT_CHUNKS_COLLECTION,
            data=[dense_vec],
            anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            filter=expr,
            output_fields=out_fields,
        )
        raw_hits = results[0] if results else []
    else:
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            expr=expr,
        )
        results = client.hybrid_search(
            collection_name=DOCUMENT_CHUNKS_COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(alpha, 1.0 - alpha),
            limit=top_k,
            output_fields=out_fields,
        )
        raw_hits = results[0] if results else []

    hits: list[dict[str, Any]] = []
    for hit in raw_hits:
        entity = hit if isinstance(hit, dict) else (hit.entity if hasattr(hit, "entity") else {})
        hits.append({
            "id":             entity.get("id"),
            "transaction_id": entity.get("transaction_id", ""),
            "file_id":        entity.get("file_id", ""),
            "document_id":    entity.get("document_id", ""),
            "doc_type":       entity.get("doc_type", ""),
            "page_numbers":   entity.get("page_numbers", "[]"),
            "chunk_type":     entity.get("chunk_type", ""),
            "chunk_index":    entity.get("chunk_index", 0),
            "chunk_text":     entity.get("chunk_text", ""),
            "chunk_metadata": entity.get("chunk_metadata", "{}"),
        })
    return hits


def search_document_chunks_multi(
    transaction_id: str,
    queries_by_type: dict[str, list[str]],
    top_k: int | None = None,
    top_t: int | None = None,
) -> list[dict[str, Any]]:
    """Multi-category multi-query hybrid search with RRF fusion.

    Parameters
    ----------
    transaction_id  : scope filter (mandatory)
    queries_by_type : dict mapping category → query strings:
                      {
                        "item":   [...],  → searches table + header chunks
                        "header": [...],  → searches header chunks only
                        "footer": [...],  → searches footer chunks only
                        "hs":     [...],  → searches table chunks only (HS-code exact)
                      }
    top_k           : ANN candidates per individual query (default: RAG_RETRIEVAL_TOP_K)
    top_t           : final chunks after RRF fusion (default: RAG_FINAL_TOP_K)

    Returns
    -------
    List of unique chunk dicts sorted by RRF score (descending), each with
    an extra ``rrf_score`` key.
    """
    from app.models.embedding import embed_text_queries  # local import

    _top_k = top_k or settings.RAG_RETRIEVAL_TOP_K
    _top_t = top_t or settings.RAG_FINAL_TOP_K

    # Map category → Milvus chunk_type filter expression
    _EXPR_MAP = {
        "item":   f'transaction_id == "{transaction_id}" && (chunk_type == "table" || chunk_type == "header")',
        "header": f'transaction_id == "{transaction_id}" && chunk_type == "header"',
        "footer": f'transaction_id == "{transaction_id}" && chunk_type == "footer"',
        "hs":     f'transaction_id == "{transaction_id}" && chunk_type == "table"',
    }

    out_fields = [
        "id", "transaction_id", "file_id", "document_id", "doc_type",
        "page_numbers", "chunk_type", "chunk_index", "chunk_text", "chunk_metadata",
    ]

    # Collect all (query_text, expr) pairs preserving association
    pairs: list[tuple[str, str]] = []
    for category, queries in queries_by_type.items():
        expr = _EXPR_MAP.get(category, f'transaction_id == "{transaction_id}"')
        for q in queries:
            if q and q.strip():
                pairs.append((q.strip(), expr))

    if not pairs:
        return []

    query_texts = [p[0] for p in pairs]

    try:
        dense_vecs, sparse_vecs = embed_text_queries(query_texts)
    except Exception as exc:
        logger.warning("search_document_chunks_multi: embedding failed — %s", exc)
        return []

    client = get_client()
    all_ranked: list[list[dict[str, Any]]] = []

    for i, ((_, expr), dense_vec, sparse_vec) in enumerate(
        zip(pairs, dense_vecs, sparse_vecs)
    ):
        try:
            hits = _run_single_hybrid_search(client, dense_vec, sparse_vec, expr, _top_k, out_fields)
            all_ranked.append(hits)
        except Exception as exc:
            logger.warning("search_document_chunks_multi: query %d failed — %s", i, exc)
            all_ranked.append([])

    # ── Reciprocal Rank Fusion (k=60) ──────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, dict[str, Any]] = {}

    for ranked_list in all_ranked:
        for rank, hit in enumerate(ranked_list, start=1):
            cid = hit.get("id")
            if not cid:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank)
            if cid not in chunk_store:
                chunk_store[cid] = hit

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    result: list[dict[str, Any]] = []
    for cid in sorted_ids[:_top_t]:
        chunk = dict(chunk_store[cid])
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        result.append(chunk)

    logger.info(
        "search_document_chunks_multi: tx=%s categories=%s total_queries=%d "
        "unique_chunks=%d returned=%d",
        transaction_id, list(queries_by_type.keys()),
        len(pairs), len(rrf_scores), len(result),
    )
    return result

