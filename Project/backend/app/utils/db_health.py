from __future__ import annotations

from fastapi import HTTPException

from app.core.logger import get_logger
from app.services import milvus_service, postgres_service

logger = get_logger(__name__)


def check_postgres() -> bool:
    return postgres_service.ping()


def check_milvus() -> bool:
    try:
        client = milvus_service.get_client()
        client.list_collections()
        return True
    except Exception:
        logger.exception("Milvus health check failed")
        return False


def assert_databases_ready() -> None:
    postgres_ok = check_postgres()
    milvus_ok = check_milvus()

    if not postgres_ok or not milvus_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Database availability check failed",
                "postgres": postgres_ok,
                "milvus": milvus_ok,
            },
        )
