from fastapi import APIRouter

from app.api.v1.endpoints import documents, status, visual_templates, transactions, coo_verification, users

api_router = APIRouter()
api_router.include_router(status.router,            prefix="/status",       tags=["status"])
api_router.include_router(documents.router,         prefix="/documents",    tags=["documents"])
api_router.include_router(
    visual_templates.router,
    prefix="/visual-templates",
    tags=["visual-templates"],
)
api_router.include_router(transactions.router,      prefix="/transactions", tags=["transactions"])
api_router.include_router(coo_verification.router,  prefix="/transactions", tags=["coo-verification"])
api_router.include_router(users.router,             prefix="/users",        tags=["users"])
