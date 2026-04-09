"""
predict.py — REST prediction endpoints.

POST /api/v1/predict/letter — single-frame letter prediction
POST /api/v1/predict/word  — accumulates frames for word prediction
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from ..main import get_arabic_coordinator, get_english_coordinator
from ..schemas.prediction_request import PredictionRequest
from ..schemas.prediction_response import PredictionResponse

router = APIRouter(tags=["predict"])


def _decode_frame(frame_b64: str) -> np.ndarray:
    """Decode a base64-encoded JPEG/PNG to an RGB numpy array."""
    raw = base64.b64decode(frame_b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_landmarks(landmarks: list[float]) -> np.ndarray:
    """Convert a list of 63 floats to a (63,) numpy array."""
    arr = np.array(landmarks, dtype=np.float32)
    if arr.shape != (63,):
        raise HTTPException(
            status_code=400,
            detail=f"Expected 63 landmarks, got {arr.shape[0]}",
        )
    return arr


@router.post("/predict/letter", response_model=PredictionResponse)
async def predict_letter(req: PredictionRequest):
    """Single-frame letter prediction (landmarks or image)."""
    coord = get_english_coordinator() if req.language == "en" else get_arabic_coordinator()
    if coord is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    if req.landmarks:
        landmarks = _decode_landmarks(req.landmarks)
        lp = coord._letter_predictor
        label, conf = lp.predict(landmarks)
        return PredictionResponse(
            prediction=label,
            confidence=round(conf, 4),
            mode="letter",
            language=req.language,
            timestamp=time.time(),
        )
    elif req.frame_b64:
        rgb = _decode_frame(req.frame_b64)
        result = coord.process_frame(rgb)
        return PredictionResponse(
            prediction=result.letter_label or "",
            confidence=round(result.letter_confidence, 4),
            mode="letter",
            language=req.language,
            timestamp=time.time(),
            decoded_text=result.decoded_text,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide landmarks or frame_b64")


@router.post("/predict/word", response_model=PredictionResponse)
async def predict_word(req: PredictionRequest):
    """Accumulate frame for word prediction."""
    coord = get_english_coordinator() if req.language == "en" else get_arabic_coordinator()
    if coord is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    if req.frame_b64:
        rgb = _decode_frame(req.frame_b64)
        result = coord.process_frame(rgb)
        return PredictionResponse(
            prediction=result.word_label or "",
            confidence=round(result.word_confidence, 4),
            mode="word",
            language=req.language,
            timestamp=time.time(),
            decoded_text=getattr(result, "sentence", ""),
        )
    else:
        raise HTTPException(status_code=400, detail="Provide frame_b64 for word prediction")
