from fastapi import FastAPI

from app.api.router import register_routes
from app.core.config import settings
from app.core.startup import lifespan

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="FastAPI project template",
    lifespan=lifespan,
)

register_routes(app)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}", "docs": "/docs"}
