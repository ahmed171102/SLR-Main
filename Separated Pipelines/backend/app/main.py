"""
main.py — FastAPI application entry point.

Provides REST + WebSocket endpoints for both English and Arabic
sign-language inference pipelines.

Run:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes import health, predict, websocket

logger = logging.getLogger("eshara")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ── Pipeline singletons (created on startup) ──
_pipelines: dict = {}


def get_pipelines() -> dict:
    """Return the shared pipeline dict.  Keys: 'en', 'ar'."""
    return _pipelines


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load models once on startup, shut down cleanly."""
    logger.info("⏳ Loading English ASL pipeline …")
    from english_pipeline.inference.pipeline import EnglishPipeline

    _pipelines["en"] = EnglishPipeline(enable_llm=True)
    logger.info("✅ English pipeline ready")

    logger.info("⏳ Loading Arabic ArSL pipeline …")
    from arabic_pipeline.inference.pipeline import ArabicPipeline

    _pipelines["ar"] = ArabicPipeline(enable_llm=True)
    logger.info("✅ Arabic pipeline ready")

    yield  # ── app is running ──

    logger.info("🛑 Shutting down pipelines")
    _pipelines.clear()


# ── Create app ──
app = FastAPI(
    title="ESHARA — Bilingual SLR API",
    version="1.0.0",
    description="Real-time sign-language recognition with separate English/Arabic pipelines and LLM correction.",
    lifespan=lifespan,
)

# ── CORS — allow frontend origins ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(websocket.router, tags=["websocket"])
