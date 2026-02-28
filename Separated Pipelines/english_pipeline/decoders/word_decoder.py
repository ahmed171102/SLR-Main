"""
word_decoder.py — English ASL word-level sentence builder.

Takes word predictions from the BiLSTM model and builds sentences,
with cooldown to prevent duplicate detections, and integration with
the LLM correction agent for low-confidence reranking.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from config.settings import (
    EN_WORD_CONFIDENCE_THRESHOLD,
    EN_WORD_COOLDOWN_S,
)


class EnglishWordDecoder:
    """Build sentences from word-level predictions.

    Features:
      - Confidence threshold: ignores predictions below threshold.
      - Cooldown: prevents the same word being added repeatedly.
      - Duplicate suppression: won't add the same word twice in a row.
    """

    def __init__(
        self,
        confidence_threshold: float = EN_WORD_CONFIDENCE_THRESHOLD,
        cooldown_s: float = EN_WORD_COOLDOWN_S,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.cooldown_s = cooldown_s

        self._words: List[str] = []
        self._word_ids: List[int] = []
        self._last_commit_ts: Optional[float] = None
        self._last_word_id: Optional[int] = None

    def reset(self) -> None:
        self._words.clear()
        self._word_ids.clear()
        self._last_commit_ts = None
        self._last_word_id = None

    @property
    def sentence(self) -> str:
        return " ".join(self._words)

    @property
    def word_count(self) -> int:
        return len(self._words)

    def update(
        self,
        word: str,
        word_id: int,
        confidence: float,
        ts: Optional[float] = None,
    ) -> Dict:
        """Feed a word prediction and potentially commit it.

        Returns
        -------
        dict with keys: committed (str), sentence (str), event (str), confidence (float)
        """
        if ts is None:
            ts = time.time()

        empty = {
            "committed": "",
            "sentence": self.sentence,
            "event": "none",
            "confidence": confidence,
        }

        # Confidence check
        if confidence < self.confidence_threshold:
            return empty

        # Cooldown
        if self._last_commit_ts is not None and (ts - self._last_commit_ts) < self.cooldown_s:
            return empty

        # Duplicate suppression
        if self._last_word_id is not None and word_id == self._last_word_id:
            return empty

        # Commit word
        self._words.append(word)
        self._word_ids.append(word_id)
        self._last_commit_ts = ts
        self._last_word_id = word_id

        return {
            "committed": word,
            "sentence": self.sentence,
            "event": "word_added",
            "confidence": confidence,
        }

    def replace_last_word(self, corrected_word: str) -> None:
        """Replace the most recently added word (after LLM correction)."""
        if self._words:
            self._words[-1] = corrected_word

    def delete_last_word(self) -> Optional[str]:
        """Remove and return the last word."""
        if self._words:
            self._word_ids.pop()
            return self._words.pop()
        return None
