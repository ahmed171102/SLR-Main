"""
word_predictor.py — English ASL word prediction from 30-frame sequences.

Loads the ASL BiLSTM model and word class/vocabulary mappings from the
original project. Input: (30, 63) sequence → Output: (word, word_id, confidence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    ASL_WORD_MODEL,
    ASL_WORD_CLASSES_CSV,
    SHARED_VOCABULARY_CSV,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
)
from shared.models.model_loader import (
    load_keras_model,
    load_word_classes,
    load_shared_vocabulary,
)


class EnglishWordPredictor:
    """Predict ASL words from a 30-frame landmark sequence.

    The model is lazily loaded on first call (or via explicit ``load()``).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        classes_csv: Optional[Path] = None,
        vocab_csv: Optional[Path] = None,
    ) -> None:
        self._model_path = model_path or ASL_WORD_MODEL
        self._classes_csv = classes_csv or ASL_WORD_CLASSES_CSV
        self._vocab_csv = vocab_csv or SHARED_VOCABULARY_CSV
        self._model = None
        self._class_to_word_id: Dict[int, int] = {}
        self._vocabulary: Dict[int, Dict[str, str]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._model = load_keras_model(self._model_path)
        self._class_to_word_id = load_word_classes(self._classes_csv)
        self._vocabulary = load_shared_vocabulary(self._vocab_csv)
        self._loaded = True

    def predict(
        self, sequence: np.ndarray
    ) -> Tuple[str, int, float]:
        """Predict a word from a (30, 63) sequence.

        Returns
        -------
        (english_word, word_id, confidence)
        """
        if not self._loaded:
            self.load()

        x = sequence.astype(np.float32).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        conf = float(probs[class_idx])

        word_id = self._class_to_word_id.get(class_idx, class_idx)
        entry = self._vocabulary.get(word_id, {})
        english = entry.get("english", f"word_{word_id}")

        return english, word_id, conf

    def predict_top_k(
        self, sequence: np.ndarray, k: int = 5
    ) -> List[Tuple[str, int, float]]:
        """Return top-k word predictions."""
        if not self._loaded:
            self.load()

        x = sequence.astype(np.float32).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:k]

        results = []
        for idx in top_indices:
            conf = float(probs[idx])
            word_id = self._class_to_word_id.get(int(idx), int(idx))
            entry = self._vocabulary.get(word_id, {})
            english = entry.get("english", f"word_{word_id}")
            results.append((english, word_id, conf))
        return results

    def get_bilingual(self, word_id: int) -> Dict[str, str]:
        """Look up both English and Arabic for a word_id."""
        if not self._loaded:
            self.load()
        return self._vocabulary.get(word_id, {"english": "?", "arabic": "?", "category": "?"})

    @property
    def vocabulary(self) -> Dict[int, Dict[str, str]]:
        if not self._loaded:
            self.load()
        return dict(self._vocabulary)
