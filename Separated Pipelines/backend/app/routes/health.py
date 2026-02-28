"""
health.py — Health-check and info endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    from backend.app.main import get_pipelines

    pipes = get_pipelines()
    return {
        "status": "ok",
        "pipelines": {
            "en": "en" in pipes,
            "ar": "ar" in pipes,
        },
    }


@router.get("/info")
async def info():
    return {
        "project": "ESHARA",
        "version": "1.0.0",
        "languages": ["en", "ar"],
        "endpoints": {
            "health": "/health",
            "predict_frame": "/api/predict",
            "reset": "/api/reset/{lang}",
            "websocket": "/ws/{lang}",
        },
    }
