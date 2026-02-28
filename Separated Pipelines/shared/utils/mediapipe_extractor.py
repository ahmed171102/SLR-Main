"""
mediapipe_extractor.py — Shared MediaPipe hand-landmark extraction.

Both English and Arabic pipelines use the EXACT same MediaPipe configuration
to extract 63 features (21 landmarks × 3 coords) per frame.
"""

from __future__ import annotations

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None  # allow import for type-checking even without mediapipe installed

from config.settings import (
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONF,
    MEDIAPIPE_MIN_TRACKING_CONF,
    MEDIAPIPE_MAX_NUM_HANDS,
    NUM_LANDMARKS,
    NUM_FEATURES,
)


class MediaPipeExtractor:
    """Extract 63-dim hand landmark features from an RGB frame.

    Thread-safety: create one instance per thread / pipeline.
    """

    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF,
            max_num_hands=MEDIAPIPE_MAX_NUM_HANDS,
        )

    # ── public API ───────────────────────────────────────────

    def extract(self, rgb_frame: np.ndarray) -> np.ndarray | None:
        """Return (63,) float32 array of landmarks, or None if no hand found.

        Parameters
        ----------
        rgb_frame : np.ndarray
            HxWx3 uint8 RGB image (NOT BGR — convert with cv2.cvtColor first).
        """
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]          # use first detected hand
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
            dtype=np.float32,
        )
        assert landmarks.shape == (NUM_LANDMARKS, 3)
        return landmarks.flatten()                       # → (63,)

    def extract_both_hands(
        self, rgb_frame: np.ndarray
    ) -> list[np.ndarray]:
        """Return list of (63,) arrays — one per detected hand (0, 1, or 2)."""
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return []
        return [
            np.array(
                [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                dtype=np.float32,
            ).flatten()
            for hand in results.multi_hand_landmarks
        ]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
