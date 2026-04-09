"""
main.py — FastAPI application entry point.

Run with:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ...config.settings import Settings

logger = logging.getLogger(__name__)

# ── Model singletons (loaded once at startup) ──────────────────────
_english_coordinator = None
_arabic_coordinator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release on shutdown."""
    global _english_coordinator, _arabic_coordinator

    from ...english_pipeline.inference.coordinator import EnglishPipelineCoordinator
    from ...arabic_pipeline.inference.coordinator import ArabicPipelineCoordinator

    logger.info("Loading English pipeline …")
    _english_coordinator = EnglishPipelineCoordinator()
    _english_coordinator.load_models()

    logger.info("Loading Arabic pipeline …")
    _arabic_coordinator = ArabicPipelineCoordinator()
    _arabic_coordinator.load_models()

    logger.info("All models loaded — server ready")
    yield

    # Shutdown
    if _english_coordinator:
        _english_coordinator.close()
    if _arabic_coordinator:
        _arabic_coordinator.close()
    logger.info("Shutdown complete")


def get_english_coordinator():
    return _english_coordinator


def get_arabic_coordinator():
    return _arabic_coordinator


app = FastAPI(
    title="ESHARA — Sign Language Recognition API",
    version="0.1.0",
    description="Real-time ASL + ArSL recognition with letter/word modes",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ──────────────────────────────────────────────
from .routes import health, predict, websocket_route  # noqa: E402

app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")
app.include_router(websocket_route.router)
