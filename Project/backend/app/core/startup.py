from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.logger import get_logger, setup_logging
from app.models.embedding import load_model
from app.models.ocr import load_ocr
from app.services.milvus_service import connect, ensure_collection

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.LOG_LEVEL)
    logger.info("Startup: loading models and connecting to Milvus")
    load_model()
    load_ocr()
    connect()
    ensure_collection()
    logger.info("Startup complete — ready to serve")
    yield
    logger.info("Shutdown")
