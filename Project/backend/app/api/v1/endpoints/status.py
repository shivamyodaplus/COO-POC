from fastapi import APIRouter

from app.core.config import settings
from app.utils.helpers import get_utc_now

router = APIRouter()


@router.get("")
async def get_status():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": get_utc_now(),
    }
