"""
letter_predictor.py — English ASL single-frame letter prediction.

Loads the ASL MLP model and label encoder from the original project.
Input: (63,) landmarks from MediaPipe → Output: (label, confidence) tuple.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config.settings import ASL_LETTER_MLP_MODEL, ASL_LETTER_DATASET_CSV, NUM_FEATURES
from shared.models.model_loader import load_keras_model, load_label_encoder_from_csv


class EnglishLetterPredictor:
    """Predict ASL fingerspelled letters from a single frame of landmarks.

    The model is lazily loaded on first call (or via explicit ``load()``).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_csv: Optional[Path] = None,
    ) -> None:
        self._model_path = model_path or ASL_LETTER_MLP_MODEL
        self._labels_csv = labels_csv or ASL_LETTER_DATASET_CSV
        self._model = None
        self._labels: List[str] = []
        self._loaded = False

    def load(self) -> None:
        """Eagerly load model + labels. Called automatically on first predict."""
        if self._loaded:
            return
        self._model = load_keras_model(self._model_path)
        self._labels = load_label_encoder_from_csv(self._labels_csv)
        self._loaded = True

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Predict a single letter from (63,) landmarks.

        Returns
        -------
        (label, confidence) : Tuple[str, float]
            e.g. ("A", 0.95)
        """
        if not self._loaded:
            self.load()

        x = landmarks.astype(np.float32).reshape(1, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self._labels[idx] if idx < len(self._labels) else str(idx)
        return label, conf

    def predict_top_k(
        self, landmarks: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k predictions sorted by descending confidence."""
        if not self._loaded:
            self.load()

        x = landmarks.astype(np.float32).reshape(1, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:k]
        return [
            (self._labels[i] if i < len(self._labels) else str(i), float(probs[i]))
            for i in top_indices
        ]

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> List[str]:
        if not self._loaded:
            self.load()
        return list(self._labels)
