"""
letter_decoder.py — Arabic ArSL letter stream decoder.

Converts per-frame Arabic letter predictions into accumulated Arabic text.
Uses Arabic-specific thresholds (slightly higher confidence, longer cooldown)
because the ArSL model is stronger (~95% val accuracy) and Arabic letters
have different spacing requirements (RTL text, connected letters).
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from config.settings import (
    AR_LETTER_CONFIDENCE_THRESHOLD,
    AR_LETTER_STABLE_WINDOW,
    AR_LETTER_MAJORITY_RATIO,
    AR_LETTER_COOLDOWN_S,
)


@dataclass
class ArabicDecoderEvent:
    committed: str       # character committed ("" if none)
    event: str           # "append" | "space" | "delete" | "none"
    text: str            # full accumulated Arabic text
    word: str            # current word (after last space)


class ArabicLetterDecoder:
    """Decode a stream of per-frame ArSL letter predictions into Arabic text.

    Arabic-specific behavior:
    - Uses Arabic control label names ("space", "del", "nothing") — same as English
    - Committed characters are Arabic unicode (ا, ب, ت…) not romanized names
    - Text is built left-to-right internally but displayed RTL by the frontend
    """

    def __init__(
        self,
        min_confidence: float = AR_LETTER_CONFIDENCE_THRESHOLD,
        stable_window: int = AR_LETTER_STABLE_WINDOW,
        majority_ratio: float = AR_LETTER_MAJORITY_RATIO,
        cooldown_s: float = AR_LETTER_COOLDOWN_S,
    ) -> None:
        self.min_confidence = min_confidence
        self.stable_window = stable_window
        self.majority_ratio = majority_ratio
        self.cooldown_s = cooldown_s

        self._labels: Deque[str] = deque(maxlen=stable_window)
        self._text_chars: List[str] = []
        self._last_commit_ts: Optional[float] = None
        self._last_committed_label: Optional[str] = None

        # Track for LLM correction
        self._recent_confidences: List[float] = []
        self._recent_labels: List[str] = []

    def reset(self) -> None:
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
        return list(zip(self._recent_labels, self._recent_confidences))

    def update(
        self,
        romanized_label: str,
        arabic_char: str,
        confidence: float,
        ts: Optional[float] = None,
    ) -> ArabicDecoderEvent:
        """Feed one frame prediction.

        Parameters
        ----------
        romanized_label : str
            Model output label (e.g. "Alef", "space", "nothing").
        arabic_char : str
            Corresponding Arabic character (e.g. "ا", " ", "").
        confidence : float
            Model softmax confidence.
        ts : float, optional
            Timestamp.
        """
        if ts is None:
            ts = time.time()

        norm = romanized_label.strip().lower() if romanized_label else ""
        self._labels.append(norm)

        empty = ArabicDecoderEvent(committed="", event="none", text=self.text, word=self.word)

        # Ignore "nothing"
        if norm == "nothing":
            return empty

        # Cooldown
        if self._last_commit_ts is not None and (ts - self._last_commit_ts) < self.cooldown_s:
            return empty

        # Need full window
        if len(self._labels) < self.stable_window:
            return empty

        # Majority voting
        counts = Counter(x for x in self._labels if x != "nothing")
        if not counts:
            return empty

        top_label, top_count = counts.most_common(1)[0]
        if (top_count / len(self._labels)) < self.majority_ratio:
            return empty

        # Confidence check
        if confidence < self.min_confidence:
            return empty

        # Avoid same-label repetition
        if self._last_committed_label is not None and top_label == self._last_committed_label:
            return empty

        # Commit
        result = self._commit(top_label, arabic_char, confidence)
        if result.event != "none":
            self._last_commit_ts = ts
            self._last_committed_label = top_label

        return result

    def _commit(self, stable_label: str, arabic_char: str, confidence: float) -> ArabicDecoderEvent:
        # Space
        if stable_label == "space":
            if self._text_chars and self._text_chars[-1] == " ":
                return ArabicDecoderEvent(committed="", event="none", text=self.text, word=self.word)
            self._text_chars.append(" ")
            return ArabicDecoderEvent(committed=" ", event="space", text=self.text, word=self.word)

        # Delete
        if stable_label == "del":
            if not self._text_chars:
                return ArabicDecoderEvent(committed="", event="none", text=self.text, word=self.word)
            self._text_chars.pop()
            return ArabicDecoderEvent(committed="", event="delete", text=self.text, word=self.word)

        # Arabic character
        ch = arabic_char if arabic_char else stable_label
        self._text_chars.append(ch)

        # Track for LLM
        self._recent_labels.append(ch)
        self._recent_confidences.append(confidence)

        return ArabicDecoderEvent(committed=ch, event="append", text=self.text, word=self.word)

    def get_current_word_for_correction(self) -> Tuple[List[str], List[float]]:
        """Get current word's Arabic letters and confidences for LLM correction."""
        word = self.word
        if not word:
            return [], []
        n = len(word)
        letters = self._recent_labels[-n:]
        confs = self._recent_confidences[-n:]
        return letters, confs

    def apply_correction(self, corrected_word: str) -> None:
        """Replace current word with LLM-corrected Arabic text."""
        current_word = self.word
        if not current_word:
            return
        for _ in range(len(current_word)):
            if self._text_chars:
                self._text_chars.pop()
        for ch in corrected_word:
            self._text_chars.append(ch)
