"""
Transactions API — lifecycle management for user transactions.

Endpoints
---------
POST   /transactions                      — create a new transaction
GET    /transactions/{tx_id}              — get a single transaction
GET    /users/{user_id}/transactions      — list all transactions for a user
GET    /transactions/{tx_id}/report       — get the latest verification report
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import transaction_service
from app.utils.db_health import assert_databases_ready

router = APIRouter()


class CreateTransactionBody(BaseModel):
    user_id: str = Field(..., min_length=1, description="Caller identity string.")
    tx_id: str | None = Field(None, min_length=1, description="Optional custom transaction ID (e.g. '1', 'order-42'). Auto-generated UUID if omitted.")


@router.post("")
async def create_transaction(body: CreateTransactionBody):
    assert_databases_ready()
    record = transaction_service.create_transaction(body.user_id.strip(), tx_id=body.tx_id)
    return record


@router.get("/{transaction_id}")
async def get_transaction(transaction_id: str):
    record = transaction_service.get_transaction(transaction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return record


@router.get("/{transaction_id}/report")
async def get_report(transaction_id: str):
    report = transaction_service.get_report(transaction_id)
    if report is None:
        raise HTTPException(status_code=404, detail="No verification report found for this transaction")
    return report
