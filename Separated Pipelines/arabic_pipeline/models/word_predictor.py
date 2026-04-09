"""
word_predictor.py — Arabic (ArSL) word prediction wrapper.

Same architecture as English word predictor (BiLSTM + TemporalAttention).
Will be functional once the ArSL word model is trained.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ArabicWordPredictor:
    """Predict Arabic words from a sequence of MediaPipe landmarks.

    Args:
        model_path: Path to ``arsl_word_lstm_model_best.h5``.
        word_classes: dict mapping model_class_index → word_id.
        vocabulary: dict mapping word_id → {english, arabic, category}.
        sequence_length: Number of frames (default 30).
        confidence_threshold: Minimum confidence (default 0.35).
    """

    def __init__(
        self,
        model_path: str | Path,
        word_classes: Dict[int, int],
        vocabulary: Dict[int, Dict[str, str]],
        sequence_length: int = 30,
        confidence_threshold: float = 0.35,
    ) -> None:
        self._model = None
        self._model_path = Path(model_path)
        self.word_classes = word_classes
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

        # Try to load model (may not exist yet)
        if self._model_path.exists():
            from ...shared.models.model_loader import load_keras_model
            self._model = load_keras_model(self._model_path)
            logger.info("ArabicWordPredictor loaded — %d classes", len(word_classes))
        else:
            logger.warning(
                "ArSL word model not found at %s — word prediction disabled. "
                "Train the model first.", self._model_path
            )

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def buffer_full(self) -> bool:
        return len(self._buffer) >= self.sequence_length

    def add_frame(self, landmarks: np.ndarray) -> None:
        self._buffer.append(landmarks.astype(np.float32))

    def clear_buffer(self) -> None:
        self._buffer.clear()

    def predict(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict Arabic word from the current buffer.

        Returns:
            List of ``(arabic_word, confidence)`` tuples.
            Empty list if model not loaded or buffer not full.
        """
        if not self.is_available or not self.buffer_full:
            return []

        seq = np.array(list(self._buffer), dtype=np.float32)
        x = seq.reshape(1, self.sequence_length, -1)
        probs = self._model.predict(x, verbose=0)[0]

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
            arabic = entry.get("arabic", f"كلمة_{word_id}")
            results.append((arabic, conf))

        return results
