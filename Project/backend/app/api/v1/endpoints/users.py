"""
Users API — user-scoped queries.

Endpoints
---------
GET /users/{user_id}/transactions  — list all transactions for a user
"""

from __future__ import annotations

from fastapi import APIRouter

from app.services import transaction_service

router = APIRouter()


@router.get("/{user_id}/transactions")
async def list_user_transactions(user_id: str):
    return transaction_service.list_user_transactions(user_id.strip())
