"""
websocket_message.py — Pydantic schemas for WebSocket protocol.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class WSIncomingFrame(BaseModel):
    """Client → Server: one video frame."""

    frame: str = Field(description="Base64-encoded JPEG frame")
    language: Optional[str] = Field(
        default=None,
        description="Override language for this frame (en | ar)",
    )


class WSOutgoingResult(BaseModel):
    """Server → Client: prediction result for one frame."""

    language: str
    mode: str
    letter_label: Optional[str] = None
    letter_confidence: float = 0.0
    word_label: Optional[str] = None
    word_confidence: float = 0.0
    decoded_text: str = ""
    decoded_text_rtl: Optional[str] = None
    current_word: str = ""
    sentence: str = ""
    sentence_rtl: Optional[str] = None
    llm_corrected: bool = False
    timestamp: float = 0.0
    error: Optional[str] = None
