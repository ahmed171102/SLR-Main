"""
word_predictor.py — Arabic ArSL word prediction from 30-frame sequences.

Loads the ArSL BiLSTM model (once KArSL-502 dataset is obtained and model
is trained).  Falls back gracefully if the model file doesn't exist yet.

Uses the SAME shared bilingual vocabulary as the English pipeline, but
returns Arabic words as primary output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    ARSL_WORD_MODEL,
    ASL_WORD_CLASSES_CSV,   # same class mapping structure (word_id based)
    SHARED_VOCABULARY_CSV,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
)
from shared.models.model_loader import (
    load_keras_model,
    load_word_classes,
    load_shared_vocabulary,
)

logger = logging.getLogger(__name__)


class ArabicWordPredictor:
    """Predict ArSL words from a 30-frame landmark sequence.

    If the Arabic word model hasn't been trained yet (KArSL-502 not
    downloaded), it logs a warning and all predictions return
    ("غير متوفر", -1, 0.0).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        classes_csv: Optional[Path] = None,
        vocab_csv: Optional[Path] = None,
    ) -> None:
        self._model_path = model_path or ARSL_WORD_MODEL
        self._classes_csv = classes_csv or ASL_WORD_CLASSES_CSV
        self._vocab_csv = vocab_csv or SHARED_VOCABULARY_CSV
        self._model = None
        self._class_to_word_id: Dict[int, int] = {}
        self._vocabulary: Dict[int, Dict[str, str]] = {}
        self._loaded = False
        self._available = False  # True only if model file exists

    def load(self) -> None:
        if self._loaded:
            return

        # Always load vocabulary (it exists)
        self._vocabulary = load_shared_vocabulary(self._vocab_csv)

        # Attempt to load the Arabic word model
        if self._model_path.exists():
            try:
                self._model = load_keras_model(self._model_path)
                self._class_to_word_id = load_word_classes(self._classes_csv)
                self._available = True
                logger.info("Arabic word model loaded: %s", self._model_path.name)
            except Exception as e:
                logger.warning("Failed to load Arabic word model: %s", e)
                self._available = False
        else:
            logger.warning(
                "Arabic word model not found at %s. "
                "Train it after downloading KArSL-502 dataset.",
                self._model_path,
            )
            self._available = False

        self._loaded = True

    @property
    def is_available(self) -> bool:
        """Whether the Arabic word model is loaded and ready."""
        if not self._loaded:
            self.load()
        return self._available

    def predict(self, sequence: np.ndarray) -> Tuple[str, int, float]:
        """Predict a word from a (30, 63) sequence.

        Returns
        -------
        (arabic_word, word_id, confidence)
        """
        if not self._loaded:
            self.load()

        if not self._available:
            return ("غير متوفر", -1, 0.0)   # "not available"

        x = sequence.astype(np.float32).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        conf = float(probs[class_idx])

        word_id = self._class_to_word_id.get(class_idx, class_idx)
        entry = self._vocabulary.get(word_id, {})
        arabic = entry.get("arabic", f"كلمة_{word_id}")

        return arabic, word_id, conf

    def predict_top_k(
        self, sequence: np.ndarray, k: int = 5
    ) -> List[Tuple[str, int, float]]:
        """Return top-k word predictions (arabic, word_id, confidence)."""
        if not self._loaded:
            self.load()

        if not self._available:
            return [("غير متوفر", -1, 0.0)]

        x = sequence.astype(np.float32).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:k]

        results = []
        for idx in top_indices:
            conf = float(probs[idx])
            word_id = self._class_to_word_id.get(int(idx), int(idx))
            entry = self._vocabulary.get(word_id, {})
            arabic = entry.get("arabic", f"كلمة_{word_id}")
            results.append((arabic, word_id, conf))
        return results

    def get_bilingual(self, word_id: int) -> Dict[str, str]:
        if not self._loaded:
            self.load()
        return self._vocabulary.get(word_id, {"english": "?", "arabic": "?", "category": "?"})
