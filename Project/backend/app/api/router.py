from fastapi import APIRouter, FastAPI

from app.api.v1.router import api_router as v1_router

_api_router = APIRouter()
_api_router.include_router(v1_router, prefix="/v1")


def register_routes(app: FastAPI) -> None:
    app.include_router(_api_router, prefix="/api")
