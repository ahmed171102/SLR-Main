"""
schemas.py — Pydantic models for request / response payloads.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    language: str = Field(..., description="'en' or 'ar'")
    mode: str = Field(..., description="letter | word | idle")
    letter_text: str = Field("", description="Decoded letter sequence / spelled text")
    word_sentence: str = Field("", description="Built sentence from word predictions")
    last_prediction: str = Field("")
    arabic_char: Optional[str] = Field(None, description="Arabic character (AR pipeline only)")
    confidence: float = 0.0
    motion: float = 0.0
    llm_corrected: bool = False


class ResetResponse(BaseModel):
    status: str = "ok"
    language: str
    message: str = ""
