from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.models.embedding import load_model
from app.models.ocr import load_ocr
from app.services.milvus_service import connect, ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Startup: loading models and connecting to Milvus ===")
    load_model()
    load_ocr()
    connect()
    ensure_collection()
    print("=== Startup complete — ready to serve ===")
    yield
    print("=== Shutdown ===")
