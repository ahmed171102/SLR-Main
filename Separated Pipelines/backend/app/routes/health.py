"""
health.py — Health check endpoint.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "eshara-slr"}


@router.get("/languages")
async def supported_languages():
    return {
        "languages": [
            {"code": "en", "name": "English (ASL)", "modes": ["letter", "word"]},
            {"code": "ar", "name": "Arabic (ArSL)", "modes": ["letter", "word"]},
        ]
    }
