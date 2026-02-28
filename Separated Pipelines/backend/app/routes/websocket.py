"""
websocket.py — Real-time WebSocket endpoint for continuous inference.

Client connects to  ws://host:port/ws/{lang}  and sends JPEG frames
as binary messages.  Server responds with JSON prediction results.
"""

from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_pipeline(lang: str):
    from backend.app.main import get_pipelines

    pipes = get_pipelines()
    return pipes.get(lang)


def _decode_bytes(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@router.websocket("/ws/{lang}")
async def ws_inference(ws: WebSocket, lang: str):
    """
    Persistent WebSocket for real-time SLR inference.

    Protocol
    --------
    Client → Server:
        Binary message: raw JPEG / PNG bytes  (preferred, lowest latency)
        Text   message: JSON {"image": "<base64>"}  (fallback)
        Text   message: JSON {"command": "reset"}   (clear decoders)

    Server → Client:
        JSON with prediction result (same schema as REST /api/predict).
    """
    pipeline = _get_pipeline(lang)
    if pipeline is None:
        await ws.close(code=4000, reason=f"Unknown language: {lang}")
        return

    await ws.accept()
    logger.info("WS connected — lang=%s", lang)

    try:
        while True:
            msg = await ws.receive()

            # ── binary frame ──
            if "bytes" in msg and msg["bytes"]:
                bgr = _decode_bytes(msg["bytes"])
                if bgr is None:
                    await ws.send_json({"error": "decode_failed"})
                    continue
                result = pipeline.process_frame(bgr)
                await ws.send_json(result)
                continue

            # ── text message (base64 or command) ──
            if "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"error": "invalid_json"})
                    continue

                # command: reset
                if payload.get("command") == "reset":
                    pipeline.reset()
                    await ws.send_json({"status": "reset", "language": lang})
                    continue

                # base64 image
                b64 = payload.get("image", "")
                if b64:
                    raw = base64.b64decode(b64)
                    bgr = _decode_bytes(raw)
                    if bgr is None:
                        await ws.send_json({"error": "decode_failed"})
                        continue
                    result = pipeline.process_frame(bgr)
                    await ws.send_json(result)
                    continue

                await ws.send_json({"error": "unknown_payload"})

    except WebSocketDisconnect:
        logger.info("WS disconnected — lang=%s", lang)
    except Exception as exc:
        logger.exception("WS error — lang=%s: %s", lang, exc)
        await ws.close(code=1011)
