"""
Milvus service for the 2-Tier GraphRAG COO verification pipeline.

Manages a single collection: ``coo_reference_chunks``

Schema
------
id               INT64            AUTO_ID primary key
item_id          VARCHAR(128)     unique chunk identifier (<txn_id>__<chunk_N>)
transaction_id   VARCHAR(64)      scalar filter — INVERTED index for O(1) lookup
source_document  VARCHAR(256)     original filename
raw_text         VARCHAR(65535)   full chunk text (no truncation)
metadata_json    VARCHAR(2048)    JSON-encoded chunk attributes
dense_vector     FLOAT_VECTOR(1024)  HNSW + COSINE
sparse_vector    SPARSE_FLOAT_VECTOR  SPARSE_INVERTED_INDEX + IP
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

COLLECTION = settings.COO_REFERENCE_COLLECTION
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


def ensure_coo_reference_chunks_collection() -> None:
    """Create (or verify) the ``coo_reference_chunks`` collection.

    Drops and recreates if the schema is incompatible.
    """
    client = get_client()

    if client.has_collection(COLLECTION):
        # Verify schema has the expected fields
        try:
            desc = client.describe_collection(COLLECTION)
            field_names = {f.get("name", "") for f in desc.get("fields", [])}
            if "item_id" in field_names and "dense_vector" in field_names:
                logger.info(
                    "Collection '%s' exists with correct schema — skipping creation",
                    COLLECTION,
                )
                return
        except Exception:
            pass
        logger.warning(
            "Collection '%s' has incompatible schema — dropping and recreating",
            COLLECTION,
        )
        client.drop_collection(COLLECTION)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("item_id", DataType.VARCHAR, max_length=128)
    schema.add_field("transaction_id", DataType.VARCHAR, max_length=64)
    schema.add_field("source_document", DataType.VARCHAR, max_length=256)
    schema.add_field("raw_text", DataType.VARCHAR, max_length=65535)
    schema.add_field("metadata_json", DataType.VARCHAR, max_length=2048)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    idx = client.prepare_index_params()
    idx.add_index(
        "dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    idx.add_index(
        "sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    idx.add_index("transaction_id", index_type="INVERTED")

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=idx,
    )
    logger.info("Created collection '%s' ✓", COLLECTION)


def insert_reference_chunks(records: list[dict[str, Any]]) -> int:
    """Insert reference chunk records into the collection.

    Each record must contain: item_id, transaction_id, source_document,
    raw_text, metadata_json, dense_vector, sparse_vector.

    Returns the number of records inserted.
    """
    if not records:
        return 0
    result = get_client().insert(collection_name=COLLECTION, data=records)
    count = result.get("insert_count", len(records))
    logger.info("Inserted %d chunks into '%s'", count, COLLECTION)
    return count


def hybrid_search_reference(
    transaction_id: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search for reference chunks within a transaction.

    Parameters
    ----------
    transaction_id : scope filter (mandatory)
    dense_vec      : 1024-dim dense query vector
    sparse_vec     : {token_id: weight} sparse query vector
    top_k          : max hits to return (default: RAG_RETRIEVAL_TOP_K)
    threshold      : minimum score to keep (default: SCORE_THRESHOLD)

    Returns list of chunk dicts sorted by score (descending).
    """
    client = get_client()
    _top_k = top_k or settings.RAG_RETRIEVAL_TOP_K
    _threshold = threshold if threshold is not None else settings.SCORE_THRESHOLD

    expr = f'transaction_id == "{transaction_id}"'
    out_fields = ["item_id", "raw_text", "metadata_json", "transaction_id"]

    alpha = settings.HYBRID_ALPHA

    if not sparse_vec:
        # Dense-only fallback
        results = client.search(
            collection_name=COLLECTION,
            data=[dense_vec],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=_top_k,
            filter=expr,
            output_fields=out_fields,
        )
        raw_hits = results[0] if results else []
    else:
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=_top_k,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=_top_k,
            expr=expr,
        )
        results = client.hybrid_search(
            collection_name=COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(alpha, 1.0 - alpha),
            limit=_top_k,
            output_fields=out_fields,
        )
        raw_hits = results[0] if results else []

    hits: list[dict[str, Any]] = []
    for hit in raw_hits:
        if isinstance(hit, dict):
            distance = hit.get("distance", 0.0)
            entity = hit
        else:
            distance = hit.distance
            entity = hit.entity if hasattr(hit, "entity") else {}

        if distance >= _threshold:
            hits.append({
                "chunk_id": entity.get("item_id"),
                "raw_text": entity.get("raw_text"),
                "metadata_json": entity.get("metadata_json"),
                "score": round(float(distance), 6),
            })

    return hits


def delete_transaction_chunks(transaction_id: str) -> None:
    """Delete all chunks for a specific transaction."""
    client = get_client()
    expr = f'transaction_id == "{transaction_id}"'
    client.delete(collection_name=COLLECTION, filter=expr)
    logger.info("Deleted chunks for transaction '%s'", transaction_id)
