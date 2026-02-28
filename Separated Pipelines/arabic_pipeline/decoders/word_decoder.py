"""
word_decoder.py — Arabic ArSL word-level sentence builder.

Builds Arabic sentences from word predictions, with cooldown,
duplicate suppression, and LLM correction integration.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from config.settings import (
    AR_WORD_CONFIDENCE_THRESHOLD,
    AR_WORD_COOLDOWN_S,
)


class ArabicWordDecoder:
    """Build Arabic sentences from word-level predictions."""

    def __init__(
        self,
        confidence_threshold: float = AR_WORD_CONFIDENCE_THRESHOLD,
        cooldown_s: float = AR_WORD_COOLDOWN_S,
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
        if ts is None:
            ts = time.time()

        empty = {
            "committed": "",
            "sentence": self.sentence,
            "event": "none",
            "confidence": confidence,
        }

        if confidence < self.confidence_threshold:
            return empty

        if self._last_commit_ts is not None and (ts - self._last_commit_ts) < self.cooldown_s:
            return empty

        if self._last_word_id is not None and word_id == self._last_word_id:
            return empty

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
        if self._words:
            self._words[-1] = corrected_word

    def delete_last_word(self) -> Optional[str]:
        if self._words:
            self._word_ids.pop()
            return self._words.pop()
        return None
