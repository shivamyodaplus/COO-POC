"""
Milvus service for the PACD / COO knowledge base.

Uses a dedicated collection ``pacd_coo_documents`` (separate from the existing
``templates`` collection) so the existing retrieval pipeline is untouched.

Schema
------
id            VARCHAR(256)  PK  — "{doc_id}_p{page_num}"
user_id        VARCHAR(128)
transaction_id VARCHAR(128)
doc_category   VARCHAR(16)   — 'pacd' | 'coo'
filename       VARCHAR(512)
page_num       INT
ocr_text       VARCHAR(65535) — analyzer + BM25 sparse
extracted_kv   VARCHAR(65535) — JSON dump of [{key, value}] pairs
dense          FLOAT_VECTOR(2048) — DINOv2 / image embedding
sparse         SPARSE_FLOAT_VECTOR — BM25 on ocr_text

Indexes: HNSW on dense, SPARSE_INVERTED_INDEX on sparse,
         INVERTED on user_id / transaction_id / doc_category.
"""

from __future__ import annotations

from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    WeightedRanker,
)

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

COLLECTION = settings.PACD_MILVUS_COLLECTION
DENSE_DIM = 2048

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    if _client is None:
        raise RuntimeError("PACD Milvus client not initialised. Call connect() first.")
    return _client


def connect() -> None:
    global _client
    _client = MilvusClient(uri=settings.MILVUS_URI)
    logger.info("PACD Milvus: connected at %s", settings.MILVUS_URI)


def ensure_pacd_collection() -> None:
    client = get_client()

    if client.has_collection(COLLECTION):
        logger.info("PACD collection '%s' already exists — skipping creation", COLLECTION)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",             DataType.VARCHAR,          is_primary=True, max_length=256)
    schema.add_field("user_id",        DataType.VARCHAR,          max_length=128)
    schema.add_field("transaction_id", DataType.VARCHAR,          max_length=128)
    schema.add_field("doc_category",   DataType.VARCHAR,          max_length=16)
    schema.add_field("filename",       DataType.VARCHAR,          max_length=512)
    schema.add_field("page_num",       DataType.INT32)
    schema.add_field("ocr_text",       DataType.VARCHAR,          max_length=65535,
                     enable_analyzer=True, enable_match=True)
    schema.add_field("extracted_kv",   DataType.VARCHAR,          max_length=65535)
    schema.add_field("dense",          DataType.FLOAT_VECTOR,     dim=DENSE_DIM)
    schema.add_field("sparse",         DataType.SPARSE_FLOAT_VECTOR)

    schema.add_function(Function(
        name="bm25_pacd",
        input_field_names=["ocr_text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    idx = client.prepare_index_params()
    idx.add_index("dense",          index_type="HNSW",                  metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    idx.add_index("sparse",         index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                  params={"drop_ratio_build": 0.2})
    idx.add_index("user_id",        index_type="INVERTED")
    idx.add_index("transaction_id", index_type="INVERTED")
    idx.add_index("doc_category",   index_type="INVERTED")

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=idx,
    )
    logger.info("PACD Milvus: created collection '%s'", COLLECTION)


def insert_pacd_pages(records: list[dict[str, Any]]) -> None:
    """Upsert page records into the PACD collection.

    Each record must contain all schema fields.  The ``extracted_kv`` field
    should be a JSON-serialised list of {key, value} dicts.
    """
    get_client().upsert(collection_name=COLLECTION, data=records)


def search_by_transaction(
    transaction_id: str,
    query_text: str,
    dense_vec: list[float],
    top_k: int = 5,
    doc_category: str = "pacd",
    extra_transaction_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search within one (or more) transaction(s).

    Parameters
    ----------
    transaction_id:
        Primary transaction to search.
    query_text:
        Free-text query (field key + value) for BM25 component.
    dense_vec:
        Dense embedding for COSINE ANN component.
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

    if not query_text or not query_text.strip():
        results = client.search(
            collection_name=COLLECTION,
            data=[dense_vec],
            anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            filter=expr,
            output_fields=["id", "user_id", "transaction_id", "doc_category",
                           "filename", "page_num", "ocr_text", "extracted_kv"],
        )
    else:
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
            expr=expr,
        )
        alpha = settings.HYBRID_ALPHA
        results = client.hybrid_search(
            collection_name=COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(alpha, 1.0 - alpha),
            limit=top_k,
            output_fields=["id", "user_id", "transaction_id", "doc_category",
                           "filename", "page_num", "ocr_text", "extracted_kv"],
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
            "extracted_kv":   entity.get("extracted_kv", "[]"),
            "similarity":     round(float(distance), 6),
        })
    return hits
