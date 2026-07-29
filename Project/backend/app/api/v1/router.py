from fastapi import APIRouter

from app.api.v1.endpoints import documents, status

api_router = APIRouter()
api_router.include_router(status.router,    prefix="/status",    tags=["status"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
