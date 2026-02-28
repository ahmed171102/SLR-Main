"""
mode_detector.py — Motion-based mode switching between LETTER and WORD modes.

Shared by both English and Arabic pipelines.
Determines whether the hand is *still* (fingerspelling a letter) or
*moving* (performing a word sign) by tracking landmark displacement
across consecutive frames.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Deque, Optional

import numpy as np

from config.settings import (
    MOTION_THRESHOLD_LETTER,
    MOTION_THRESHOLD_WORD,
    MOTION_BUFFER_FRAMES,
)


class SignMode(str, Enum):
    LETTER = "letter"
    WORD = "word"
    IDLE = "idle"          # no hand detected


class ModeDetector:
    """Detect whether the signer is fingerspelling or performing a word sign.

    Strategy (Option A from integration doc — motion-based):
    - Compute mean absolute displacement of all 63 features between consecutive frames.
    - Smooth over ``buffer_frames`` to avoid jitter.
    - If smoothed motion < ``letter_threshold`` → LETTER mode (hand still).
    - If smoothed motion > ``word_threshold``   → WORD mode (hand moving).
    - In between → keep previous mode (hysteresis band prevents flapping).
    """

    def __init__(
        self,
        letter_threshold: float = MOTION_THRESHOLD_LETTER,
        word_threshold: float = MOTION_THRESHOLD_WORD,
        buffer_frames: int = MOTION_BUFFER_FRAMES,
    ) -> None:
        self._letter_thresh = letter_threshold
        self._word_thresh = word_threshold
        self._buffer: Deque[float] = deque(maxlen=max(1, buffer_frames))
        self._prev_landmarks: Optional[np.ndarray] = None
        self._current_mode: SignMode = SignMode.IDLE

    # ── public API ───────────────────────────────────────────

    def update(self, landmarks: Optional[np.ndarray]) -> SignMode:
        """Feed a (63,) landmark array and get the current mode.

        Parameters
        ----------
        landmarks : (63,) float32 or None
            Current frame landmarks.  Pass None when no hand is detected.

        Returns
        -------
        SignMode
            LETTER, WORD, or IDLE.
        """
        if landmarks is None:
            self._prev_landmarks = None
            self._buffer.clear()
            self._current_mode = SignMode.IDLE
            return SignMode.IDLE

        if self._prev_landmarks is not None:
            displacement = float(np.mean(np.abs(landmarks - self._prev_landmarks)))
            self._buffer.append(displacement)

        self._prev_landmarks = landmarks.copy()

        if len(self._buffer) == 0:
            return self._current_mode

        smoothed = float(np.mean(self._buffer))

        if smoothed < self._letter_thresh:
            self._current_mode = SignMode.LETTER
        elif smoothed > self._word_thresh:
            self._current_mode = SignMode.WORD
        # else: keep previous mode (hysteresis)

        return self._current_mode

    @property
    def mode(self) -> SignMode:
        return self._current_mode

    @property
    def last_motion(self) -> float:
        """Return last smoothed motion value (for debug display)."""
        if len(self._buffer) == 0:
            return 0.0
        return float(np.mean(self._buffer))

    def reset(self) -> None:
        self._prev_landmarks = None
        self._buffer.clear()
        self._current_mode = SignMode.IDLE
