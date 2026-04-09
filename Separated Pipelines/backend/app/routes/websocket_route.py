"""
websocket_route.py — WebSocket real-time streaming endpoint.

Usage:
    ws://host:8000/ws/recognize?language=en
    ws://host:8000/ws/recognize?language=ar

Client sends: base64-encoded JPEG frames
Server replies: JSON prediction results per frame
"""

from __future__ import annotations

import base64
import json
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..main import get_arabic_coordinator, get_english_coordinator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/recognize")
async def websocket_recognize(
    ws: WebSocket,
    language: str = Query(default="en", regex="^(en|ar)$"),
):
    """Real-time streaming WebSocket endpoint.

    Protocol:
      Client → Server: JSON  { "frame": "<base64-jpeg>" }
      Server → Client: JSON  { "prediction": ..., "confidence": ..., "mode": ..., ... }
    """
    await ws.accept()
    logger.info("WebSocket connected — language=%s", language)

    coord = get_english_coordinator() if language == "en" else get_arabic_coordinator()
    if coord is None:
        await ws.send_json({"error": "Models not loaded"})
        await ws.close()
        return

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_json({"error": "Invalid JSON"})
                continue

            frame_b64 = msg.get("frame")
            if not frame_b64:
                await ws.send_json({"error": "Missing 'frame' field"})
                continue

            try:
                raw = base64.b64decode(frame_b64)
                buf = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if bgr is None:
                    await ws.send_json({"error": "Could not decode image"})
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                await ws.send_json({"error": f"Frame decode error: {e}"})
                continue

            result = coord.process_frame(rgb)

            response = {
                "language": language,
                "mode": result.mode,
                "letter_label": result.letter_label,
                "letter_confidence": round(result.letter_confidence, 4),
                "word_label": result.word_label,
                "word_confidence": round(result.word_confidence, 4),
                "decoded_text": result.decoded_text,
                "current_word": getattr(result, "current_word", ""),
                "sentence": getattr(result, "sentence", ""),
                "llm_corrected": result.llm_corrected,
                "timestamp": time.time(),
            }

            # Add RTL fields for Arabic
            if language == "ar":
                response["decoded_text_rtl"] = getattr(result, "decoded_text_rtl", "")
                response["sentence_rtl"] = getattr(result, "sentence_rtl", "")

            await ws.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — language=%s", language)
    except Exception:
        logger.exception("WebSocket error")
        await ws.close()
