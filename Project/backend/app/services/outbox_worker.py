from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from app.services import milvus_service, postgres_service

logger = get_logger(__name__)


async def run_outbox_worker() -> None:
    if settings.STORAGE_ADAPTER.lower() != "postgres":
        logger.info("Outbox worker disabled because storage adapter is '%s'", settings.STORAGE_ADAPTER)
        return

    logger.info("Outbox worker started")
    poll_interval = max(1.0, float(settings.OUTBOX_POLL_INTERVAL))

    while True:
        try:
            rows = await asyncio.to_thread(postgres_service.claim_outbox_batch, 20)
            if not rows:
                await asyncio.sleep(poll_interval)
                continue

            for row in rows:
                await _process_outbox_row(row)
        except asyncio.CancelledError:
            logger.info("Outbox worker stopping")
            raise
        except Exception:
            logger.exception("Outbox worker iteration failed")
            await asyncio.sleep(poll_interval)


async def _process_outbox_row(row: dict[str, Any]) -> None:
    outbox_id = int(row["id"])
    entity_id = str(row["entity_id"])
    retry_count = int(row.get("retry_count", 0))

    try:
        payload = row.get("payload") or {}
        operation = str(row.get("operation") or "UPSERT").upper()

        if operation in {"UPSERT", "CREATE", "UPDATE"}:
            record = {
                "id": payload["id"],
                "country": payload["country"],
                "doc_type": payload["doc_type"],
                "file_name": payload["file_name"],
                "ocr_text": payload.get("ocr_text", ""),
                "dense": payload["dense"],
            }
            await asyncio.to_thread(milvus_service.insert, [record])
        elif operation == "DELETE":
            client = milvus_service.get_client()
            await asyncio.to_thread(
                client.delete,
                collection_name=milvus_service.COLLECTION,
                ids=[entity_id],
            )
        else:
            raise ValueError(f"Unsupported outbox operation: {operation}")

        await asyncio.to_thread(postgres_service.mark_outbox_processed, outbox_id, entity_id)
    except Exception as exc:
        logger.exception("Outbox row %s failed: %s", outbox_id, exc)
        await asyncio.to_thread(
            postgres_service.mark_outbox_failed_or_retry,
            outbox_id,
            retry_count,
            str(exc),
        )
