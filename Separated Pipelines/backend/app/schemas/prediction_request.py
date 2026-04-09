"""
prediction_request.py — Pydantic schema for prediction input.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input payload for /predict/letter and /predict/word endpoints.

    Provide EITHER:
      * landmarks: list of 63 floats (pre-extracted MediaPipe landmarks)
      * frame_b64: base64-encoded JPEG/PNG image (server runs MediaPipe)
    """

    language: str = Field(
        default="en",
        pattern="^(en|ar)$",
        description="Language: 'en' for ASL, 'ar' for ArSL",
    )
    mode: str = Field(
        default="auto",
        pattern="^(auto|letter|word)$",
        description="Recognition mode: auto, letter, or word",
    )
    landmarks: Optional[List[float]] = Field(
        default=None,
        description="21 MediaPipe hand landmarks flattened to 63 floats (x,y,z per joint)",
    )
    frame_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded JPEG or PNG image frame",
    )
