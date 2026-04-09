"""
prediction_response.py — Pydantic schema for prediction output.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Output payload from /predict/letter and /predict/word."""

    prediction: str = Field(description="Predicted label (letter or word)")
    confidence: float = Field(description="Model confidence [0, 1]")
    mode: str = Field(description="Recognition mode used: letter | word | idle")
    language: str = Field(description="Language: en | ar")
    timestamp: float = Field(description="Server timestamp (epoch seconds)")
    decoded_text: Optional[str] = Field(
        default=None,
        description="Accumulated decoded text from the letter decoder",
    )
    decoded_text_rtl: Optional[str] = Field(
        default=None,
        description="RTL-formatted decoded text (Arabic only)",
    )
    sentence: Optional[str] = Field(
        default=None,
        description="Accumulated sentence from word decoder",
    )
    llm_corrected: bool = Field(
        default=False,
        description="Whether LLM correction was applied",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra info (vocab entry, correction details, etc.)",
    )
