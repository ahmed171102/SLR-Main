"""
mediapipe_extractor.py — Extract hand landmarks using MediaPipe Hands.

Provides a reusable class that takes an RGB frame and returns
a 63-dimensional float32 array (21 landmarks × 3 coords).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("MediaPipe is required: pip install mediapipe") from exc


class MediaPipeExtractor:
    """Extract hand landmarks from camera frames using MediaPipe Hands.

    Usage::

        extractor = MediaPipeExtractor()
        landmarks = extractor.extract(rgb_frame)   # (63,) or None
        extractor.close()

    Or as a context manager::

        with MediaPipeExtractor() as ext:
            landmarks = ext.extract(rgb_frame)
    """

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        max_num_hands: int = 2,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands,
        )

    # ── Context manager ──────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    # ── Extraction ───────────────────────────────────

    def extract(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract landmarks for the dominant (first detected) hand.

        Args:
            rgb_frame: H×W×3 uint8 RGB image.

        Returns:
            ``(63,)`` float32 array  or  ``None`` if no hand detected.
        """
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        return np.array(coords, dtype=np.float32)

    def extract_both_hands(self, rgb_frame: np.ndarray) -> List[np.ndarray]:
        """Extract landmarks for all detected hands (0 – max_num_hands).

        Args:
            rgb_frame: H×W×3 uint8 RGB image.

        Returns:
            List of ``(63,)`` float32 arrays (may be empty).
        """
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return []

        hands_out = []
        for hand in results.multi_hand_landmarks:
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            hands_out.append(np.array(coords, dtype=np.float32))

        return hands_out
