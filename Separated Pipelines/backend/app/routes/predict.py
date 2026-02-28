"""
predict.py — REST endpoints for single-frame prediction and reset.
"""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from backend.app.schemas import PredictionResponse, ResetResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_pipeline(lang: str):
    from backend.app.main import get_pipelines

    pipes = get_pipelines()
    if lang not in pipes:
        raise HTTPException(status_code=400, detail=f"Unknown language: {lang}")
    return pipes[lang]


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw bytes → BGR numpy image."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image")
    return img


# ──────────────────────────────────────────────
# POST /api/predict  — upload a single frame
# ──────────────────────────────────────────────
@router.post("/predict", response_model=PredictionResponse)
async def predict_frame(
    lang: str = Form("en", description="'en' or 'ar'"),
    frame: UploadFile = File(..., description="Camera frame (JPEG / PNG)"),
):
    """Process one camera frame through the selected pipeline."""
    pipeline = _get_pipeline(lang)
    raw = await frame.read()
    bgr = _decode_image(raw)
    result = pipeline.process_frame(bgr)
    return PredictionResponse(**result)


# ──────────────────────────────────────────────
# POST /api/predict/base64  — JSON with base64 image
# ──────────────────────────────────────────────
@router.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(payload: dict):
    """Accept {"lang": "en"|"ar", "image": "<base64>"}."""
    lang = payload.get("lang", "en")
    b64 = payload.get("image", "")
    if not b64:
        raise HTTPException(status_code=422, detail="Missing 'image' field")

    pipeline = _get_pipeline(lang)
    raw = base64.b64decode(b64)
    bgr = _decode_image(raw)
    result = pipeline.process_frame(bgr)
    return PredictionResponse(**result)


# ──────────────────────────────────────────────
# POST /api/reset/{lang}
# ──────────────────────────────────────────────
@router.post("/reset/{lang}", response_model=ResetResponse)
async def reset_pipeline(lang: str):
    """Clear decoder state for the specified language."""
    pipeline = _get_pipeline(lang)
    pipeline.reset()
    return ResetResponse(language=lang, message=f"{lang.upper()} pipeline reset")
