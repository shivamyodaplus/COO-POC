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

            # ── Extend existing tables (idempotent) ─────────────────────
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS transaction_id TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_category TEXT NOT NULL DEFAULT 'pacd'"
            )
            cur.execute(
                "ALTER TABLE visual_templates ADD COLUMN IF NOT EXISTS extracted_attributes JSONB"
            )
            cur.execute(
                "ALTER TABLE visual_templates ADD COLUMN IF NOT EXISTS page_num INT NOT NULL DEFAULT 0"
            )
            cur.execute(
                "ALTER TABLE visual_templates ADD COLUMN IF NOT EXISTS attributes_status TEXT NOT NULL DEFAULT 'pending'"
            )
            cur.execute(
                "ALTER TABLE visual_templates ADD COLUMN IF NOT EXISTS doc_category TEXT NOT NULL DEFAULT 'any'"
            )

            # ── Transaction management ───────────────────────────────────
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # Migrate existing UUID column to TEXT (idempotent — no-op if already TEXT)
            cur.execute(
                """
                DO $$ BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='transactions' AND column_name='id'
                          AND data_type='uuid'
                    ) THEN
                        ALTER TABLE verification_reports
                            DROP CONSTRAINT IF EXISTS verification_reports_transaction_id_fkey;
                        ALTER TABLE transactions ALTER COLUMN id TYPE TEXT;
                        ALTER TABLE verification_reports ALTER COLUMN transaction_id TYPE TEXT;
                        ALTER TABLE verification_reports
                            ADD CONSTRAINT verification_reports_transaction_id_fkey
                            FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE;
                    END IF;
                END $$;
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions (user_id)"
            )

            # ── Verification reports ─────────────────────────────────────
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_reports (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    transaction_id TEXT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
                    coo_doc_id TEXT NOT NULL DEFAULT '',
                    confirmed_template_id TEXT,
                    confirmed_template_name TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    discrepancy_table JSONB NOT NULL DEFAULT '[]',
                    narrative TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_verification_reports_tx ON verification_reports (transaction_id)"
            )

            # ── Structured PACD storage ──────────────────────────────────
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pacd_documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    transaction_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    doc_type TEXT NOT NULL DEFAULT 'other',
                    filename TEXT NOT NULL DEFAULT '',
                    page_numbers INTEGER[] NOT NULL DEFAULT '{}',
                    header_fields JSONB NOT NULL DEFAULT '{}',
                    raw_ocr_text TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pacd_docs_transaction ON pacd_documents (transaction_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pacd_docs_user ON pacd_documents (user_id)"
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pacd_line_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pacd_document_id UUID NOT NULL REFERENCES pacd_documents(id) ON DELETE CASCADE,
                    transaction_id TEXT NOT NULL,
                    line_number INTEGER,
                    hs_code TEXT,
                    hs_code_normalized TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    quantity TEXT,
                    unit TEXT,
                    unit_price TEXT,
                    total_value TEXT,
                    weight TEXT,
                    origin_country TEXT,
                    raw_text TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pacd_items_transaction ON pacd_line_items (transaction_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pacd_items_hs ON pacd_line_items (hs_code_normalized)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pacd_items_doc ON pacd_line_items (pacd_document_id)"
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

                # Only enqueue a Milvus outbox entry when a dense vector is
                # present — records without one (e.g. PACD pages whose Milvus
                # indexing is handled by pacd_milvus_service) must be skipped.
                dense = item.get("dense")
                if dense:
                    payload = {
                        "id": item["id"],
                        "country": item["country"],
                        "doc_type": item["doc_type"],
                        "doc_category": item.get("doc_category", ""),
                        "file_name": item["file_name"],
                        "ocr_text": item["ocr_text"],
                        "dense": dense,
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


# ──────────────────────────────────────────────────────────────────────────────
# Structured PACD Document CRUD
# ──────────────────────────────────────────────────────────────────────────────


def insert_pacd_document(
    transaction_id: str,
    user_id: str,
    doc_type: str,
    filename: str,
    page_numbers: list[int],
    header_fields: dict[str, str],
    raw_ocr_text: str = "",
) -> str:
    """Insert a logical PACD document and return its UUID."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pacd_documents
                    (transaction_id, user_id, doc_type, filename, page_numbers, header_fields, raw_ocr_text)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING id
                """,
                (
                    transaction_id,
                    user_id,
                    doc_type,
                    filename,
                    page_numbers,
                    json.dumps(header_fields),
                    raw_ocr_text[:100000],
                ),
            )
            doc_id = str(cur.fetchone()[0])
        conn.commit()
    return doc_id


def insert_pacd_line_items(
    pacd_document_id: str,
    transaction_id: str,
    items: list[dict[str, Any]],
) -> list[str]:
    """Insert line items for a PACD document. Returns list of created UUIDs."""
    if not items:
        return []
    pool = get_pool()
    item_ids: list[str] = []
    with pool.connection() as conn:
        with conn.cursor() as cur:
            for item in items:
                hs_raw = item.get("hs_code") or ""
                # Normalize HS code: strip dots/spaces, take first 6 digits
                hs_normalized = hs_raw.replace(".", "").replace(" ", "")[:6] if hs_raw else None
                cur.execute(
                    """
                    INSERT INTO pacd_line_items
                        (pacd_document_id, transaction_id, line_number, hs_code,
                         hs_code_normalized, description, quantity, unit,
                         unit_price, total_value, weight, origin_country, raw_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        pacd_document_id,
                        transaction_id,
                        item.get("line_number") or item.get("item_number"),
                        hs_raw or None,
                        hs_normalized,
                        item.get("description", ""),
                        item.get("quantity"),
                        item.get("unit"),
                        item.get("unit_price"),
                        item.get("total_value"),
                        item.get("weight"),
                        item.get("origin_country"),
                        item.get("raw_text"),
                    ),
                )
                item_ids.append(str(cur.fetchone()[0]))
        conn.commit()
    return item_ids


def get_pacd_documents_by_transaction(transaction_id: str) -> list[dict[str, Any]]:
    """Fetch all PACD documents (with header fields) for a transaction."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, transaction_id, user_id, doc_type, filename,
                       page_numbers, header_fields, created_at
                FROM pacd_documents
                WHERE transaction_id = %s
                ORDER BY created_at
                """,
                (transaction_id,),
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in (cur.description or [])]
    results = []
    for row in rows:
        item = dict(zip(columns, row))
        # Ensure header_fields is a dict
        hf = item.get("header_fields")
        if isinstance(hf, str):
            item["header_fields"] = json.loads(hf)
        item["id"] = str(item["id"])
        results.append(item)
    return results


def get_pacd_line_items_by_transaction(
    transaction_id: str,
    hs_code_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch PACD line items for a transaction, optionally filtered by HS code prefix."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if hs_code_prefix:
                # Normalize the search prefix too
                normalized_prefix = hs_code_prefix.replace(".", "").replace(" ", "")[:6]
                cur.execute(
                    """
                    SELECT li.*, pd.filename AS source_filename, pd.doc_type AS source_doc_type
                    FROM pacd_line_items li
                    JOIN pacd_documents pd ON pd.id = li.pacd_document_id
                    WHERE li.transaction_id = %s
                      AND li.hs_code_normalized LIKE %s
                    ORDER BY li.line_number
                    """,
                    (transaction_id, f"{normalized_prefix}%"),
                )
            else:
                cur.execute(
                    """
                    SELECT li.*, pd.filename AS source_filename, pd.doc_type AS source_doc_type
                    FROM pacd_line_items li
                    JOIN pacd_documents pd ON pd.id = li.pacd_document_id
                    WHERE li.transaction_id = %s
                    ORDER BY li.line_number
                    """,
                    (transaction_id,),
                )
            rows = cur.fetchall()
            columns = [desc[0] for desc in (cur.description or [])]
    results = []
    for row in rows:
        item = dict(zip(columns, row))
        item["id"] = str(item["id"])
        item["pacd_document_id"] = str(item["pacd_document_id"])
        results.append(item)
    return results


def get_pacd_line_items_by_ids(item_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch specific PACD line items by their UUIDs."""
    if not item_ids:
        return []
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(item_ids))
            cur.execute(
                f"""
                SELECT li.*, pd.filename AS source_filename, pd.doc_type AS source_doc_type
                FROM pacd_line_items li
                JOIN pacd_documents pd ON pd.id = li.pacd_document_id
                WHERE li.id::text IN ({placeholders})
                """,
                item_ids,
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in (cur.description or [])]
    results = []
    for row in rows:
        item = dict(zip(columns, row))
        item["id"] = str(item["id"])
        item["pacd_document_id"] = str(item["pacd_document_id"])
        results.append(item)
    return results
