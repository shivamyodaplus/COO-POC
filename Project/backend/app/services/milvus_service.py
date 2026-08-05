from __future__ import annotations

from typing import Any

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, WeightedRanker

from app.core.config import settings
from app.core.logger import get_logger

_client: MilvusClient | None = None
COLLECTION = settings.MILVUS_COLLECTION
DENSE_DIM = 2048
logger = get_logger(__name__)


def get_client() -> MilvusClient:
    if _client is None:
        raise RuntimeError("Milvus client not initialized. Call connect() first.")
    return _client


def connect() -> None:
    global _client
    _client = MilvusClient(uri=settings.MILVUS_URI)
    logger.info("Connected to Milvus at %s", settings.MILVUS_URI)


def _dense_dim_matches(client: MilvusClient) -> bool:
    """Return True if the existing collection's dense field has the expected dim."""
    try:
        desc = client.describe_collection(COLLECTION)
        for field in desc.get("fields", []):
            if field.get("name") == "dense":
                params = field.get("params", {})
                return int(params.get("dim", 0)) == DENSE_DIM
    except Exception:
        pass
    return False


def ensure_collection() -> None:
    client = get_client()

    if client.has_collection(COLLECTION):
        if _dense_dim_matches(client):
            logger.info("Collection '%s' already exists — skipping creation", COLLECTION)
            return
        logger.warning(
            "Collection '%s' has wrong dense dim (expected %d) — dropping and recreating",
            COLLECTION, DENSE_DIM,
        )
        client.drop_collection(COLLECTION)

    # Build schema
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",        DataType.VARCHAR,         is_primary=True, max_length=256)
    schema.add_field("country",   DataType.VARCHAR,         max_length=128)
    schema.add_field("doc_type",     DataType.VARCHAR,      max_length=128)
    schema.add_field("doc_category", DataType.VARCHAR,      max_length=16)
    schema.add_field("file_name",    DataType.VARCHAR,      max_length=512)
    schema.add_field("ocr_text",  DataType.VARCHAR,         max_length=65535,
                 enable_analyzer=True, enable_match=True)
    schema.add_field("dense",     DataType.FLOAT_VECTOR,    dim=DENSE_DIM)
    schema.add_field("sparse",    DataType.SPARSE_FLOAT_VECTOR)

    # Server-side BM25: Milvus generates sparse vectors from ocr_text automatically
    schema.add_function(Function(
        name="bm25",
        input_field_names=["ocr_text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    # Build indexes
    idx = client.prepare_index_params()
    idx.add_index("dense",    index_type="HNSW",                  metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    idx.add_index("sparse",   index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                  params={"drop_ratio_build": 0.2})
    idx.add_index("country",      index_type="INVERTED")
    idx.add_index("doc_type",     index_type="INVERTED")
    idx.add_index("doc_category", index_type="INVERTED")

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=idx,
    )
    logger.info("Created Milvus collection '%s'", COLLECTION)


def insert(records: list[dict[str, Any]]) -> None:
    client = get_client()
    client.upsert(collection_name=COLLECTION, data=records)


def hybrid_search(
    dense_vec: list[float],
    ocr_text: str,
    country: str | None,
    doc_type: str | None,
    top_k: int = 3,
    alpha: float | None = None,
    doc_category: str | None = None,
) -> list[dict[str, Any]]:
    from app.core.config import settings as cfg

    client = get_client()

    filters: list[str] = []
    if country and country.strip():
        filters.append(f'country == "{country.strip()}"')
    if doc_type and doc_type.strip():
        filters.append(f'doc_type == "{doc_type.strip()}"')
    if doc_category and doc_category.strip():
        filters.append(f'doc_category == "{doc_category.strip()}"')
    expr = " && ".join(filters) if filters else ""

    # When OCR text is empty, BM25 produces a zero sparse vector and
    # WeightedRanker returns no results — fall back to pure dense ANN search.
    if not ocr_text or not ocr_text.strip():
        results = client.search(
            collection_name=COLLECTION,
            data=[dense_vec],
            anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            filter=expr or None,
            output_fields=["id", "country", "doc_type", "file_name", "ocr_text"],
        )
    else:
        weight = alpha if alpha is not None else cfg.HYBRID_ALPHA
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[ocr_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
            expr=expr or None,
        )
        results = client.hybrid_search(
            collection_name=COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(weight, 1.0 - weight),
            limit=top_k,
            output_fields=["id", "country", "doc_type", "file_name", "ocr_text"],
        )

    hits = []
    for hit in (results[0] if results else []):
        # Handle both dict and Hit-object return formats across pymilvus versions
        if isinstance(hit, dict):
            hit_id   = hit.get("id")
            distance = hit.get("distance", 0.0)
            entity   = hit
        else:
            hit_id   = hit.id
            distance = hit.distance
            entity   = hit.entity if hasattr(hit, "entity") else {}

        hits.append({
            "id":              hit_id,
            "country":         entity.get("country"),
            "doc_type":        entity.get("doc_type"),
            "file_name":       entity.get("file_name"),
            "ocr_text_preview": (entity.get("ocr_text") or "")[:120],
            "similarity":      round(float(distance), 6),
        })
    return hits
