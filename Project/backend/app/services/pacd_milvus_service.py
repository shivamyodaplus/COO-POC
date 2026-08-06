"""
Milvus service for PACD line item description search (fuzzy fallback).

Schema (v3 — reduced role: fuzzy item matching only)
-----------------------------------------------------
The primary PACD data now lives in PostgreSQL (pacd_documents + pacd_line_items).
Milvus only indexes line item descriptions for semantic fallback when exact HS
code matching fails.

id                VARCHAR(128)  PK  — pacd_line_items.id (UUID)
transaction_id    VARCHAR(128)
hs_code           VARCHAR(20)       — raw HS code (for hybrid filter)
description_text  VARCHAR(2048)     — "{hs_code} | {description}"
dense             FLOAT_VECTOR(1024) — BGE-M3 dense embedding
sparse            SPARSE_FLOAT_VECTOR — BGE-M3 lexical weights

Indexes: HNSW on dense (COSINE), SPARSE_INVERTED_INDEX on sparse (IP),
         INVERTED on transaction_id.
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
    top_k: int = 5,
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

