"""
letter_predictor.py — English (ASL) letter prediction wrapper.

Wraps the MLP model trained on 63-dim MediaPipe landmarks.
Input: (1, 63) → Output: (label, confidence)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class EnglishLetterPredictor:
    """Predict ASL letters from MediaPipe hand landmarks.

    Args:
        model_path: Path to ``asl_mediapipe_mlp_model.h5``.
        labels: Sorted list of class label strings.
        confidence_threshold: Minimum confidence to return a prediction.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels: List[str],
        confidence_threshold: float = 0.80,
    ) -> None:
        from ...shared.models.model_loader import load_keras_model

        self.model = load_keras_model(model_path)
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        logger.info(
            "EnglishLetterPredictor ready — %d classes, threshold=%.2f",
            len(labels),
            confidence_threshold,
        )

    def predict(self, landmarks: np.ndarray) -> Optional[Tuple[str, float]]:
        """Predict letter from a single frame's landmarks.

        Args:
            landmarks: ``(63,)`` float32 array of hand landmarks.

        Returns:
            ``(label, confidence)`` or ``None`` if below threshold.
        """
        x = landmarks.reshape(1, -1).astype(np.float32)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf < self.confidence_threshold:
            return None

        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        return label, conf

    def predict_top_k(self, landmarks: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k predictions sorted by confidence."""
        x = landmarks.reshape(1, -1).astype(np.float32)
        probs = self.model.predict(x, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:k]

        results = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            results.append((label, float(probs[idx])))
        return results
