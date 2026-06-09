"""VendorWatch FastAPI application entry point."""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import dashboard, flags, vendors
from src.config import get_settings
from src.detection.model_registry import init_registry

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title       = "VendorWatch",
    description = "Supply chain anomaly detection — 3-layer ML pipeline",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(vendors.router)
app.include_router(flags.router)
app.include_router(dashboard.router)


@app.on_event("startup")
async def startup() -> None:
    registry = init_registry(settings.MODELS_DIR)
    await registry.load_all()
    # Background task: poll for model file updates every 60 s
    asyncio.create_task(registry.poll_for_updates())
    log.info("VendorWatch backend started.")
