from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg_pool import ConnectionPool

from app.adapters.base import StorageAdapter
from app.adapters.local import LocalStorageAdapter
from app.adapters.postgres import PostgresStorageAdapter
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
_pool: ConnectionPool | None = None
_local_adapter: LocalStorageAdapter | None = None


def connect() -> None:
    global _pool

    if settings.STORAGE_ADAPTER.lower() != "postgres":
        logger.info("Storage adapter is '%s'; skipping postgres pool init", settings.STORAGE_ADAPTER)
        return

    if _pool is not None:
        return

    _pool = ConnectionPool(
        conninfo=settings.POSTGRES_DSN,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": False},
    )
    _pool.open(wait=True)
    logger.info("Connected to PostgreSQL")


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("PostgreSQL pool not initialized. Call connect() first.")
    return _pool


def ping() -> bool:
    if settings.STORAGE_ADAPTER.lower() != "postgres":
        return True

    try:
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        logger.exception("PostgreSQL ping failed")
        return False


def ensure_schema() -> None:
    if settings.STORAGE_ADAPTER.lower() != "postgres":
        return

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    country TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    ocr_text TEXT NOT NULL DEFAULT '',
                    image_data BYTEA NOT NULL,
                    image_content_type TEXT NOT NULL DEFAULT 'image/jpeg',
                    milvus_synced BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS visual_templates (
                    id UUID PRIMARY KEY,
                    name TEXT NOT NULL,
                    template_type TEXT NOT NULL,
                    country TEXT,
                    doc_type TEXT,
                    image_data BYTEA NOT NULL,
                    image_content_type TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS milvus_outbox (
                    id BIGSERIAL PRIMARY KEY,
                    entity_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    operation TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error_msg TEXT,
                    next_retry_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    processed_at TIMESTAMPTZ
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_milvus_outbox_status_next_retry
                ON milvus_outbox (status, next_retry_at, id)
                """
            )
        conn.commit()


def get_adapter() -> StorageAdapter:
    global _local_adapter

    if settings.STORAGE_ADAPTER.lower() == "postgres":
        return PostgresStorageAdapter(get_pool())

    if _local_adapter is None:
        _local_adapter = LocalStorageAdapter()
    return _local_adapter


def insert_documents_with_outbox(items: list[dict[str, Any]]) -> None:
    if settings.STORAGE_ADAPTER.lower() != "postgres":
        raise RuntimeError("insert_documents_with_outbox requires postgres adapter")

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute(
                    """
                    INSERT INTO documents (
                        id,
                        country,
                        doc_type,
                        file_name,
                        ocr_text,
                        image_data,
                        image_content_type,
                        milvus_synced
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
                    ON CONFLICT (id) DO UPDATE
                    SET country = EXCLUDED.country,
                        doc_type = EXCLUDED.doc_type,
                        file_name = EXCLUDED.file_name,
                        ocr_text = EXCLUDED.ocr_text,
                        image_data = EXCLUDED.image_data,
                        image_content_type = EXCLUDED.image_content_type,
                        milvus_synced = FALSE
                    """,
                    (
                        item["id"],
                        item["country"],
                        item["doc_type"],
                        item["file_name"],
                        item["ocr_text"],
                        item["image_data"],
                        item.get("image_content_type", "image/jpeg"),
                    ),
                )

                payload = {
                    "id": item["id"],
                    "country": item["country"],
                    "doc_type": item["doc_type"],
                    "file_name": item["file_name"],
                    "ocr_text": item["ocr_text"],
                    "dense": item["dense"],
                }
                cur.execute(
                    """
                    INSERT INTO milvus_outbox (
                        entity_id,
                        operation,
                        payload,
                        status,
                        retry_count,
                        next_retry_at
                    ) VALUES (%s, %s, %s::jsonb, 'PENDING', 0, now())
                    """,
                    (item["id"], "UPSERT", json.dumps(payload)),
                )
        conn.commit()


def claim_outbox_batch(limit: int = 20) -> list[dict[str, Any]]:
    if settings.STORAGE_ADAPTER.lower() != "postgres":
        return []

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH picked AS (
                    SELECT id
                    FROM milvus_outbox
                    WHERE status = 'PENDING'
                      AND next_retry_at <= now()
                    ORDER BY id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE milvus_outbox o
                SET status = 'PROCESSING'
                FROM picked
                WHERE o.id = picked.id
                RETURNING o.id, o.entity_id, o.operation, o.payload, o.retry_count
                """,
                (limit,),
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in (cur.description or [])]
        conn.commit()

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(zip(columns, row))
        payload = item.get("payload")
        if isinstance(payload, str):
            try:
                item["payload"] = json.loads(payload)
            except json.JSONDecodeError:
                item["payload"] = {}
        out.append(item)
    return out


def mark_outbox_processed(outbox_id: int, entity_id: str) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE milvus_outbox
                SET status = 'PROCESSED',
                    processed_at = now(),
                    error_msg = NULL
                WHERE id = %s
                """,
                (outbox_id,),
            )
            cur.execute(
                """
                UPDATE documents
                SET milvus_synced = TRUE
                WHERE id = %s
                """,
                (entity_id,),
            )
        conn.commit()


def mark_outbox_failed_or_retry(
    outbox_id: int,
    retry_count: int,
    error_msg: str,
) -> None:
    pool = get_pool()
    next_retry = datetime.now(UTC) + timedelta(seconds=2 ** max(0, retry_count))
    should_fail = retry_count + 1 >= settings.OUTBOX_MAX_RETRIES

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE milvus_outbox
                SET status = %s,
                    retry_count = retry_count + 1,
                    error_msg = %s,
                    next_retry_at = %s
                WHERE id = %s
                """,
                (
                    "FAILED" if should_fail else "PENDING",
                    error_msg[:2000],
                    next_retry,
                    outbox_id,
                ),
            )
        conn.commit()
