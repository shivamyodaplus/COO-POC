"""
Milvus service for the PACD / COO knowledge base.

Schema (v2 — BGE-M3 text embeddings)
--------------------------------------
id             VARCHAR(256)  PK  — "{doc_id}_p{page}_c{chunk_idx}"
user_id        VARCHAR(128)
transaction_id VARCHAR(128)
doc_category   VARCHAR(16)   — 'pacd' | 'coo'
filename       VARCHAR(512)
page_num       INT32
chunk_index    INT32         — position of chunk within the page
chunk_text     VARCHAR(4096) — text content of this chunk
dense          FLOAT_VECTOR(1024) — BGE-M3 dense embedding
sparse         SPARSE_FLOAT_VECTOR — BGE-M3 lexical weights (client-side)

Indexes: HNSW on dense (COSINE), SPARSE_INVERTED_INDEX on sparse (IP),
         INVERTED on user_id / transaction_id / doc_category.

Breaking change vs v1: dim changed 2048 → 1024 and BM25 server function
removed.  The collection is automatically dropped and recreated when the old
schema is detected (chunk_text field absent).
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

COLLECTION = settings.PACD_MILVUS_COLLECTION
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


def _has_new_schema(client: MilvusClient) -> bool:
    """Return True if the collection uses the v2 BGE-M3 schema."""
    try:
        desc = client.describe_collection(COLLECTION)
        field_names = {f.get("name", "") for f in desc.get("fields", [])}
        return "chunk_text" in field_names
    except Exception:
        return False


def ensure_pacd_collection() -> None:
    client = get_client()

    if client.has_collection(COLLECTION):
        if _has_new_schema(client):
            logger.info("PACD collection '%s' exists with v2 schema — skipping creation", COLLECTION)
            return
        logger.warning(
            "PACD collection '%s' has old schema (v1) — dropping and recreating with v2", COLLECTION
        )
        client.drop_collection(COLLECTION)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",             DataType.VARCHAR,      is_primary=True, max_length=256)
    schema.add_field("user_id",        DataType.VARCHAR,      max_length=128)
    schema.add_field("transaction_id", DataType.VARCHAR,      max_length=128)
    schema.add_field("doc_category",   DataType.VARCHAR,      max_length=16)
    schema.add_field("filename",       DataType.VARCHAR,      max_length=512)
    schema.add_field("page_num",       DataType.INT32)
    schema.add_field("chunk_index",    DataType.INT32)
    schema.add_field("chunk_text",     DataType.VARCHAR,      max_length=4096)
    schema.add_field("dense",          DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse",         DataType.SPARSE_FLOAT_VECTOR)

    idx = client.prepare_index_params()
    idx.add_index("dense",          index_type="HNSW",                  metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    idx.add_index("sparse",         index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    idx.add_index("user_id",        index_type="INVERTED")
    idx.add_index("transaction_id", index_type="INVERTED")
    idx.add_index("doc_category",   index_type="INVERTED")

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=idx,
    )
    logger.info("PACD Milvus: created collection '%s' (v2 BGE-M3 schema)", COLLECTION)


def insert_pacd_chunks(records: list[dict[str, Any]]) -> None:
    """Upsert chunk records into the PACD collection.

    Each record must contain all schema fields.
    """
    get_client().upsert(collection_name=COLLECTION, data=records)


def search_by_transaction(
    transaction_id: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    top_k: int = 2,
    doc_category: str = "pacd",
    extra_transaction_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search within one (or more) transaction(s).

    Parameters
    ----------
    transaction_id:
        Primary transaction to search.
    dense_vec:
        BGE-M3 1024-dim dense embedding of the query.
    sparse_vec:
        BGE-M3 lexical weights dict {token_id: weight}.
    top_k:
        Number of results to return.
    doc_category:
        Filter by 'pacd' (default) or 'coo'.
    extra_transaction_ids:
        Additional transaction IDs to include (for full-history scope).
    """
    client = get_client()

    all_tx_ids = [transaction_id]
    if extra_transaction_ids:
        all_tx_ids.extend(extra_transaction_ids)

    tx_filter_parts = " || ".join(f'transaction_id == "{tid}"' for tid in all_tx_ids)
    expr = f'doc_category == "{doc_category}" && ({tx_filter_parts})'

    # Diagnostic: log total collection size and whether the filter matches anything
    try:
        total = client.get_collection_stats(COLLECTION).get("row_count", "?")
        tx_count = client.query(
            collection_name=COLLECTION,
            filter=expr,
            output_fields=["id"],
            limit=1,
        )
        logger.info(
            "PACD search: collection_rows=%s, filter_match≥%d, expr=%s",
            total, len(tx_count), expr,
        )
    except Exception:
        pass  # diagnostic only — never block the actual search

    out_fields = ["id", "user_id", "transaction_id", "doc_category",
                  "filename", "page_num", "chunk_index", "chunk_text"]

    # Fall back to pure dense search if sparse vec is empty
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
            "id":             entity.get("id"),
            "user_id":        entity.get("user_id"),
            "transaction_id": entity.get("transaction_id"),
            "doc_category":   entity.get("doc_category"),
            "filename":       entity.get("filename"),
            "page_num":       entity.get("page_num"),
            "chunk_index":    entity.get("chunk_index"),
            "chunk_text":     entity.get("chunk_text", ""),
            "similarity":     round(float(distance), 6),
        })
    return hits

