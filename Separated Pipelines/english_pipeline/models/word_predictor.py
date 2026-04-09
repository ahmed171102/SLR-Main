"""
word_predictor.py — English (ASL) word prediction wrapper.

Wraps the BiLSTM + TemporalAttention model.
Manages a 30-frame sliding buffer.
Input: (1, 30, 63) → Output: top-k (word, confidence) pairs.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class EnglishWordPredictor:
    """Predict ASL words from a sequence of MediaPipe landmarks.

    Internally manages a deque of frames. Once the buffer reaches
    ``sequence_length``, predictions can be made.

    Args:
        model_path: Path to ``asl_word_lstm_model_best.h5``.
        word_classes: dict mapping model_class_index → word_id.
        vocabulary: dict mapping word_id → {english, arabic, category}.
        sequence_length: Number of frames in the input sequence (default 30).
        confidence_threshold: Minimum confidence to return a prediction.
    """

    def __init__(
        self,
        model_path: str | Path,
        word_classes: Dict[int, int],
        vocabulary: Dict[int, Dict[str, str]],
        sequence_length: int = 30,
        confidence_threshold: float = 0.35,
    ) -> None:
        from ...shared.models.model_loader import load_keras_model

        self.model = load_keras_model(model_path)
        self.word_classes = word_classes
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
        logger.info(
            "EnglishWordPredictor ready — %d word classes, seq_len=%d",
            len(word_classes),
            sequence_length,
        )

    @property
    def buffer_full(self) -> bool:
        return len(self._buffer) >= self.sequence_length

    def add_frame(self, landmarks: np.ndarray) -> None:
        """Add a single frame to the sliding buffer."""
        self._buffer.append(landmarks.astype(np.float32))

    def clear_buffer(self) -> None:
        """Clear the frame buffer."""
        self._buffer.clear()

    def predict(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict word from the current buffer.

        Returns:
            List of ``(english_word, confidence)`` tuples, sorted descending.
            Empty list if buffer is not full or all below threshold.
        """
        if not self.buffer_full:
            return []

        seq = np.array(list(self._buffer), dtype=np.float32)
        x = seq.reshape(1, self.sequence_length, -1)
        probs = self.model.predict(x, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            conf = float(probs[idx])
            if conf < self.confidence_threshold:
                continue
            word_id = self.word_classes.get(idx)
            if word_id is None:
                continue
            entry = self.vocabulary.get(word_id, {})
            english = entry.get("english", f"word_{word_id}")
            results.append((english, conf))

        return results

    def predict_with_arabic(self, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Predict word, returning both English and Arabic translations.

        Returns:
            List of ``(english, arabic, confidence)`` tuples.
        """
        if not self.buffer_full:
            return []

        seq = np.array(list(self._buffer), dtype=np.float32)
        x = seq.reshape(1, self.sequence_length, -1)
        probs = self.model.predict(x, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            conf = float(probs[idx])
            if conf < self.confidence_threshold:
                continue
            word_id = self.word_classes.get(idx)
            if word_id is None:
                continue
            entry = self.vocabulary.get(word_id, {})
            english = entry.get("english", f"word_{word_id}")
            arabic = entry.get("arabic", "")
            results.append((english, arabic, conf))

        return results
