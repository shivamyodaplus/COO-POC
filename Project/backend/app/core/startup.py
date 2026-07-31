from __future__ import annotations

import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.logger import get_logger, setup_logging
from app.models.embedding import load_model
from app.models.ocr import load_ocr
from app.services.milvus_service import connect, ensure_collection
from app.services.outbox_worker import run_outbox_worker
from app.services.postgres_service import close as postgres_close
from app.services.postgres_service import connect as postgres_connect
from app.services.postgres_service import ensure_schema

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.LOG_LEVEL)
    logger.info("Startup: loading models and connecting databases")
    load_model()
    load_ocr()
    connect()
    ensure_collection()
    postgres_connect()
    ensure_schema()
    worker_task = asyncio.create_task(run_outbox_worker())
    logger.info("Startup complete — ready to serve")
    yield
    worker_task.cancel()
    with suppress(asyncio.CancelledError):
        await worker_task
    postgres_close()
    logger.info("Shutdown")
