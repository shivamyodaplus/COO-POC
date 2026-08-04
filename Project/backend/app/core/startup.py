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
from app.utils.template_matching import load_lightglue_models
from app.services.outbox_worker import run_outbox_worker
from app.services.postgres_service import close as postgres_close
from app.services.postgres_service import connect as postgres_connect
from app.services.postgres_service import ensure_schema
from app.services.vllm_service import check_health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.LOG_LEVEL)
    logger.info("Startup: loading models and connecting databases")
    load_model()
    load_ocr()
    load_lightglue_models()
    connect()
    ensure_collection()
    postgres_connect()
    ensure_schema()
    worker_task = asyncio.create_task(run_outbox_worker())

    # Check vLLM endpoints — non-fatal: the app starts regardless.
    vllm_status = await check_health()
    for name, healthy in vllm_status.items():
        if healthy:
            logger.info("vLLM endpoint ready: %s", name)
        else:
            logger.warning(
                "vLLM endpoint unreachable: %s — vision/text features will be unavailable. "
                "Start the servers with: bash scripts/vllm/serve_all.sh",
                name,
            )

    logger.info("Startup complete — ready to serve")
    yield
    worker_task.cancel()
    with suppress(asyncio.CancelledError):
        await worker_task
    postgres_close()
    logger.info("Shutdown")
