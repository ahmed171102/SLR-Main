"""
letter_decoder.py — English ASL letter stream decoder.

Adapted from the original ``letter_stream_decoder.py`` in SLR Main/Letters/Guides/
with English-specific thresholds from config.settings.

Converts per-frame letter predictions into accumulated text using:
  - Stability window (majority voting)
  - Cooldown to avoid repeated commits
  - Control labels: space, del, nothing
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from config.settings import (
    EN_LETTER_CONFIDENCE_THRESHOLD,
    EN_LETTER_STABLE_WINDOW,
    EN_LETTER_MAJORITY_RATIO,
    EN_LETTER_COOLDOWN_S,
)


@dataclass
class DecoderEvent:
    committed: str       # character committed ("" if none)
    event: str           # "append" | "space" | "delete" | "none"
    text: str            # full accumulated text
    word: str            # current word (after last space)


class EnglishLetterDecoder:
    """Decode a stream of per-frame ASL letter predictions into English text.

    Parameters match the original letter_stream_decoder.py but use
    English-specific defaults from config.settings.
    """

    def __init__(
        self,
        min_confidence: float = EN_LETTER_CONFIDENCE_THRESHOLD,
        stable_window: int = EN_LETTER_STABLE_WINDOW,
        majority_ratio: float = EN_LETTER_MAJORITY_RATIO,
        cooldown_s: float = EN_LETTER_COOLDOWN_S,
    ) -> None:
        self.min_confidence = min_confidence
        self.stable_window = stable_window
        self.majority_ratio = majority_ratio
        self.cooldown_s = cooldown_s

        self._labels: Deque[str] = deque(maxlen=stable_window)
        self._text_chars: List[str] = []
        self._last_commit_ts: Optional[float] = None
        self._last_committed_label: Optional[str] = None

        # Track confidences for LLM agent integration
        self._recent_confidences: List[float] = []
        self._recent_labels: List[str] = []

    def reset(self) -> None:
        """Clear all state."""
        self._labels.clear()
        self._text_chars.clear()
        self._last_commit_ts = None
        self._last_committed_label = None
        self._recent_confidences.clear()
        self._recent_labels.clear()

    @property
    def text(self) -> str:
        return "".join(self._text_chars)

    @property
    def word(self) -> str:
        txt = self.text
        if not txt or txt.endswith(" "):
            return ""
        return txt.rsplit(" ", 1)[-1]

    @property
    def recent_letters_with_confidence(self) -> List[Tuple[str, float]]:
        """Get recent committed letters with their confidences (for LLM agent)."""
        return list(zip(self._recent_labels, self._recent_confidences))

    def update(
        self, label: str, confidence: float, ts: Optional[float] = None
    ) -> DecoderEvent:
        """Feed one frame prediction and get the decode result.

        Parameters
        ----------
        label : str
            Predicted label (e.g. "A", "space", "del", "nothing").
        confidence : float
            Model softmax confidence for this label.
        ts : float, optional
            Timestamp (seconds). Defaults to time.time().

        Returns
        -------
        DecoderEvent
        """
        if ts is None:
            ts = time.time()

        norm = label.strip().lower() if label else ""
        self._labels.append(norm)

        empty = DecoderEvent(committed="", event="none", text=self.text, word=self.word)

        # Ignore "nothing"
        if norm == "nothing":
            return empty

        # Cooldown
        if self._last_commit_ts is not None and (ts - self._last_commit_ts) < self.cooldown_s:
            return empty

        # Need full window
        if len(self._labels) < self.stable_window:
            return empty

        # Majority voting (exclude "nothing")
        counts = Counter(x for x in self._labels if x != "nothing")
        if not counts:
            return empty

        top_label, top_count = counts.most_common(1)[0]
        if (top_count / len(self._labels)) < self.majority_ratio:
            return empty

        # Confidence check
        if confidence < self.min_confidence:
            return empty

        # Avoid same-letter repetition
        if self._last_committed_label is not None and top_label == self._last_committed_label:
            return empty

        # Commit
        result = self._commit(top_label, confidence)
        if result.event != "none":
            self._last_commit_ts = ts
            self._last_committed_label = top_label

        return result

    def _commit(self, stable_label: str, confidence: float) -> DecoderEvent:
        # Space
        if stable_label == "space":
            if self._text_chars and self._text_chars[-1] == " ":
                return DecoderEvent(committed="", event="none", text=self.text, word=self.word)
            self._text_chars.append(" ")
            return DecoderEvent(committed=" ", event="space", text=self.text, word=self.word)

        # Delete
        if stable_label == "del":
            if not self._text_chars:
                return DecoderEvent(committed="", event="none", text=self.text, word=self.word)
            self._text_chars.pop()
            return DecoderEvent(committed="", event="delete", text=self.text, word=self.word)

        # Letter
        ch = stable_label.upper() if len(stable_label) == 1 and stable_label.isalpha() else stable_label
        self._text_chars.append(ch)

        # Track for LLM agent
        self._recent_labels.append(ch)
        self._recent_confidences.append(confidence)

        return DecoderEvent(committed=ch, event="append", text=self.text, word=self.word)

    def get_current_word_for_correction(self) -> Tuple[List[str], List[float]]:
        """Return the current word's letters and confidences for LLM correction."""
        word = self.word
        if not word:
            return [], []
        # Get the last N letters matching current word length
        n = len(word)
        letters = self._recent_labels[-n:]
        confs = self._recent_confidences[-n:]
        return letters, confs

    def apply_correction(self, corrected_word: str) -> None:
        """Replace the current in-progress word with LLM-corrected version."""
        current_word = self.word
        if not current_word:
            return
        # Remove current word characters
        for _ in range(len(current_word)):
            if self._text_chars:
                self._text_chars.pop()
        # Add corrected characters
        for ch in corrected_word:
            self._text_chars.append(ch)
