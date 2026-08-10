"""
Transaction service — manages transaction lifecycle and verification reports.

All persistence goes directly through the Postgres pool (not the StorageAdapter,
which is scoped to document/template blob storage).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.logger import get_logger
from app.services.postgres_service import get_pool

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Transactions                                                                 #
# --------------------------------------------------------------------------- #

def create_transaction(user_id: str, tx_id: str | None = None) -> dict[str, Any]:
    """Create a new transaction and return its record.

    Args:
        user_id: Owner of the transaction.
        tx_id: Optional custom string ID. Falls back to a UUID if not provided.
    """
    pool = get_pool()
    tx_id = tx_id.strip() if tx_id else str(uuid.uuid4())
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transactions (id, user_id, status)
                VALUES (%s, %s, 'active')
                RETURNING id, user_id, status, created_at
                """,
                (tx_id, user_id),
            )
            row = cur.fetchone()
        conn.commit()

    return {
        "id": str(row[0]),
        "user_id": row[1],
        "status": row[2],
        "created_at": row[3].isoformat(),
    }


def get_transaction(transaction_id: str) -> dict[str, Any] | None:
    """Return a transaction record by ID or None if not found."""
    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, status, created_at, updated_at FROM transactions WHERE id = %s",
            (transaction_id,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": str(row[0]),
        "user_id": row[1],
        "status": row[2],
        "created_at": row[3].isoformat(),
        "updated_at": row[4].isoformat(),
    }


def list_user_transactions(user_id: str) -> list[dict[str, Any]]:
    """Return all transactions for a given user, newest first."""
    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, status, created_at, updated_at
            FROM transactions
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]),
            "user_id": r[1],
            "status": r[2],
            "created_at": r[3].isoformat(),
            "updated_at": r[4].isoformat(),
        }
        for r in rows
    ]


def update_transaction_status(transaction_id: str, status: str) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE transactions
                SET status = %s, updated_at = now()
                WHERE id = %s
                """,
                (status, transaction_id),
            )
        conn.commit()


# --------------------------------------------------------------------------- #
# Verification reports                                                         #
# --------------------------------------------------------------------------- #

def save_report(
    transaction_id: str,
    coo_doc_id: str,
    report_data: dict[str, Any],
) -> str:
    """Persist a verification report and return its UUID."""
    pool = get_pool()
    report_id = str(uuid.uuid4())

    discrepancy_table = report_data.get("discrepancy_table", [])
    # Ensure it's serialisable — each item may be a Pydantic model or a dict
    if discrepancy_table and hasattr(discrepancy_table[0], "model_dump"):
        discrepancy_table = [item.model_dump() for item in discrepancy_table]

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO verification_reports (
                    id,
                    transaction_id,
                    coo_doc_id,
                    confirmed_template_id,
                    confirmed_template_name,
                    status,
                    discrepancy_table,
                    narrative,
                    summary
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                (
                    report_id,
                    transaction_id,
                    coo_doc_id,
                    report_data.get("confirmed_template_id"),
                    report_data.get("confirmed_template_name"),
                    report_data.get("overall_verdict", "INCONCLUSIVE"),
                    json.dumps(discrepancy_table),
                    report_data.get("narrative", ""),
                    report_data.get("summary", ""),
                ),
            )
        conn.commit()

    logger.info("Saved verification report %s for transaction %s", report_id, transaction_id)
    return report_id


def get_report(transaction_id: str) -> dict[str, Any] | None:
    """Return the latest verification report for a transaction."""
    pool = get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id, transaction_id, coo_doc_id,
                confirmed_template_id, confirmed_template_name,
                status, discrepancy_table, narrative, summary, created_at
            FROM verification_reports
            WHERE transaction_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (transaction_id,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    discrepancy_table = row[6]
    if isinstance(discrepancy_table, str):
        try:
            discrepancy_table = json.loads(discrepancy_table)
        except json.JSONDecodeError:
            discrepancy_table = []

    matched_count = sum(1 for r in discrepancy_table if r.get("verdict") == "match")
    mismatched_count = sum(1 for r in discrepancy_table if r.get("verdict") == "mismatch")

    return {
        "id": str(row[0]),
        "transaction_id": str(row[1]),
        "coo_doc_id": row[2],
        "confirmed_template_id": row[3],
        "confirmed_template_name": row[4],
        "overall_verdict": row[5],
        "discrepancy_table": discrepancy_table,
        "total_fields": len(discrepancy_table),
        "matched": matched_count,
        "mismatched": mismatched_count,
        "not_found": len(discrepancy_table) - matched_count - mismatched_count,
        "narrative": row[7],
        "summary": row[8],
        "created_at": row[9].isoformat(),
    }
