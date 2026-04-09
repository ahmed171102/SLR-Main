"""
mode_detector.py — Detect whether the signer is fingerspelling (LETTER)
or performing a word-level sign (WORD) based on landmark motion.

Uses mean absolute displacement between consecutive frames, smoothed
over a configurable buffer window with a hysteresis band.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SignMode(enum.Enum):
    """Current signing mode."""
    IDLE = "idle"
    LETTER = "letter"
    WORD = "word"


class ModeDetector:
    """Classify signing mode from per-frame landmark motion.

    Strategy:
      - Track mean absolute landmark displacement per frame.
      - Smooth over ``buffer_frames`` frames to reduce jitter.
      - Below ``letter_threshold``  →  LETTER mode.
      - Above ``word_threshold``    →  WORD   mode.
      - In between                  →  keep previous mode (hysteresis).

    Args:
        letter_threshold: Motion below this = LETTER (static sign).
        word_threshold: Motion above this = WORD (dynamic sign).
        buffer_frames: Number of frames to smooth over.
    """

    def __init__(
        self,
        letter_threshold: float = 0.015,
        word_threshold: float = 0.030,
        buffer_frames: int = 5,
    ) -> None:
        if letter_threshold >= word_threshold:
            raise ValueError("letter_threshold must be < word_threshold")

        self.letter_threshold = letter_threshold
        self.word_threshold = word_threshold
        self.buffer_frames = buffer_frames

        self._motion_buffer: deque[float] = deque(maxlen=buffer_frames)
        self._prev_landmarks: Optional[np.ndarray] = None
        self._current_mode: SignMode = SignMode.IDLE

    def reset(self) -> None:
        """Reset detector state."""
        self._motion_buffer.clear()
        self._prev_landmarks = None
        self._current_mode = SignMode.IDLE

    @property
    def mode(self) -> SignMode:
        return self._current_mode

    def update(self, landmarks: Optional[np.ndarray]) -> SignMode:
        """Update with new frame landmarks and return current mode.

        Args:
            landmarks: ``(63,)`` float32 array or ``None`` (no hand).

        Returns:
            Current ``SignMode``.
        """
        if landmarks is None:
            self._prev_landmarks = None
            self._current_mode = SignMode.IDLE
            return self._current_mode

        if self._prev_landmarks is not None:
            displacement = float(np.mean(np.abs(landmarks - self._prev_landmarks)))
            self._motion_buffer.append(displacement)

        self._prev_landmarks = landmarks.copy()

        # Need enough frames to decide
        if len(self._motion_buffer) < self.buffer_frames:
            return self._current_mode

        avg_motion = float(np.mean(self._motion_buffer))

        if avg_motion < self.letter_threshold:
            self._current_mode = SignMode.LETTER
        elif avg_motion > self.word_threshold:
            self._current_mode = SignMode.WORD
        # else: hysteresis — keep previous mode

        return self._current_mode
