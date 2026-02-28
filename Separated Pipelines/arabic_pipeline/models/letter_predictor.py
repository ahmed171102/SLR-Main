"""
letter_predictor.py — Arabic ArSL single-frame letter prediction.

Loads the ArSL MLP model and label encoder from the original project.
Input: (63,) landmarks → Output: (romanized_label, arabic_char, confidence).

The model was trained with romanized label names (Alef, Beh, Teh…).
This predictor maps them to actual Arabic characters for display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import ARSL_LETTER_MLP_MODEL, ARSL_LETTER_DATASET_CSV, NUM_FEATURES
from shared.models.model_loader import load_keras_model, load_label_encoder_from_csv


# Romanized label → Arabic unicode character mapping
# Matches the Arabic Combined Notebook's NAME_TO_ARABIC dict
NAME_TO_ARABIC: Dict[str, Optional[str]] = {
    "Alef": "ا",    "Beh": "ب",    "Teh": "ت",    "Theh": "ث",
    "Jeem": "ج",    "Hah": "ح",    "Khah": "خ",    "Dal": "د",
    "Thal": "ذ",    "Reh": "ر",    "Zain": "ز",    "Seen": "س",
    "Sheen": "ش",   "Sad": "ص",    "Dad": "ض",    "Tah": "ط",
    "Zah": "ظ",     "Ain": "ع",    "Ghain": "غ",   "Feh": "ف",
    "Qaf": "ق",     "Kaf": "ك",    "Lam": "ل",    "Meem": "م",
    "Noon": "ن",    "Heh": "ه",    "Waw": "و",    "Yeh": "ي",
    # Extended set (some models)
    "Al": "ال",     "Laa": "لا",   "Teh_Marbuta": "ة",
    # Control labels
    "space": " ",   "del": "del",  "nothing": None,
}


class ArabicLetterPredictor:
    """Predict ArSL fingerspelled letters from a single frame of landmarks.

    Returns both romanized labels (for internal logic) and Arabic characters
    (for display).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_csv: Optional[Path] = None,
    ) -> None:
        self._model_path = model_path or ARSL_LETTER_MLP_MODEL
        self._labels_csv = labels_csv or ARSL_LETTER_DATASET_CSV
        self._model = None
        self._labels: List[str] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._model = load_keras_model(self._model_path)
        self._labels = load_label_encoder_from_csv(self._labels_csv)
        self._loaded = True

    def predict(self, landmarks: np.ndarray) -> Tuple[str, str, float]:
        """Predict a single Arabic letter.

        Returns
        -------
        (romanized_label, arabic_char, confidence)
            e.g. ("Alef", "ا", 0.95) or ("space", " ", 0.88)
        """
        if not self._loaded:
            self.load()

        x = landmarks.astype(np.float32).reshape(1, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        rom_label = self._labels[idx] if idx < len(self._labels) else str(idx)
        arabic = NAME_TO_ARABIC.get(rom_label, rom_label)
        return rom_label, arabic if arabic else "", conf

    def predict_top_k(
        self, landmarks: np.ndarray, k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Return top-k predictions: (romanized, arabic, confidence)."""
        if not self._loaded:
            self.load()

        x = landmarks.astype(np.float32).reshape(1, NUM_FEATURES)
        probs = self._model.predict(x, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:k]
        results = []
        for i in top_indices:
            rom = self._labels[i] if i < len(self._labels) else str(i)
            arabic = NAME_TO_ARABIC.get(rom, rom)
            results.append((rom, arabic if arabic else "", float(probs[i])))
        return results

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> List[str]:
        if not self._loaded:
            self.load()
        return list(self._labels)
