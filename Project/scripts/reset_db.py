#!/usr/bin/env python3
"""
Reset all storages to a blank state (Postgres + both Milvus collections).

Truncates every Postgres table and drops+recreates both Milvus collections
using the same schema-setup functions that run at app startup.  No Docker
rebuild required.

Usage (inside the running app container):
    docker exec <app-container> python scripts/reset_db.py

Usage (local, from Project/backend/):
    PYTHONPATH=. python ../scripts/reset_db.py
"""
from __future__ import annotations

import os
import sys

# Allow running from repo root, Project/, or Project/backend/
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(_here, "..", "backend"),   # Project/scripts/ → Project/backend/
    os.path.join(_here, "backend"),         # Project/ → Project/backend/
    _here,                                  # already inside backend/
):
    if os.path.isdir(os.path.join(_candidate, "app")):
        sys.path.insert(0, os.path.abspath(_candidate))
        break


def reset_postgres() -> None:
    print("→  Resetting PostgreSQL…")
    from app.services import postgres_service  # noqa: PLC0415

    postgres_service.connect()
    pool = postgres_service.get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Order respects FK constraints: children before parents
            cur.execute(
                """
                TRUNCATE TABLE
                    verification_reports,
                    milvus_outbox,
                    pacd_line_items,
                    pacd_documents,
                    documents,
                    visual_templates,
                    transactions
                RESTART IDENTITY CASCADE
                """
            )
        conn.commit()
    print("   ✓ All Postgres tables truncated.")


def reset_milvus() -> None:
    print("→  Resetting Milvus…")
    from app.services import milvus_service, pacd_milvus_service  # noqa: PLC0415

    # ── OCR / document-template collection ──────────────────────────────────
    milvus_service.connect()
    client = milvus_service.get_client()
    if client.has_collection(milvus_service.COLLECTION):
        client.drop_collection(milvus_service.COLLECTION)
        print(f"   ✓ Dropped   '{milvus_service.COLLECTION}'")
    milvus_service.ensure_collection()
    print(f"   ✓ Recreated '{milvus_service.COLLECTION}'")

    # ── PACD / COO reference chunks collection (GraphRAG) ──────────────────
    pacd_milvus_service.connect()
    pacd_client = pacd_milvus_service.get_client()
    if pacd_client.has_collection(pacd_milvus_service.COLLECTION):
        pacd_client.drop_collection(pacd_milvus_service.COLLECTION)
        print(f"   ✓ Dropped   '{pacd_milvus_service.COLLECTION}'")
    pacd_milvus_service.ensure_coo_reference_chunks_collection()
    print(f"   ✓ Recreated '{pacd_milvus_service.COLLECTION}'")


if __name__ == "__main__":
    reset_postgres()
    reset_milvus()
    print("\n✅  All storages reset to blank state.")
