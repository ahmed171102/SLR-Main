"""
letter_decoder.py — English letter stream decoder.

Ported from Letters/Guides/letter_stream_decoder.py with the same logic:
  - Stable window (5 frames)
  - Majority voting (70%)
  - Cooldown (0.6s)
  - Control labels: space, del, nothing
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class _CommitResult:
    committed: str
    event: str  # "append" | "space" | "delete" | "none"


class EnglishLetterDecoder:
    """Decode per-frame ASL letter predictions into accumulated text.

    Args:
        min_confidence: Minimum confidence to commit.
        stable_window: Window size for stability check.
        majority_ratio: Fraction of window that must agree.
        cooldown_s: Seconds between commits.
        control_labels: Tuple of (space, del, nothing) label strings.
    """

    def __init__(
        self,
        min_confidence: float = 0.80,
        stable_window: int = 5,
        majority_ratio: float = 0.70,
        cooldown_s: float = 0.6,
        control_labels: Tuple[str, str, str] = ("space", "del", "nothing"),
    ) -> None:
        self.min_confidence = min_confidence
        self.stable_window = stable_window
        self.majority_ratio = majority_ratio
        self.cooldown_s = cooldown_s

        self._space_label = control_labels[0].lower()
        self._del_label = control_labels[1].lower()
        self._nothing_label = control_labels[2].lower()

        self._labels: Deque[str] = deque(maxlen=stable_window)
        self._text_chars: list[str] = []
        self._last_commit_ts: Optional[float] = None
        self._last_committed_label: Optional[str] = None

    def reset(self) -> None:
        """Clear all state."""
        self._labels.clear()
        self._text_chars.clear()
        self._last_commit_ts = None
        self._last_committed_label = None

    @property
    def text(self) -> str:
        return "".join(self._text_chars)

    @property
    def word(self) -> str:
        txt = self.text
        if not txt or txt.endswith(" "):
            return ""
        return txt.rsplit(" ", 1)[-1]

    def update(self, label: str, confidence: float, ts: Optional[float] = None) -> dict:
        """Process one frame prediction.

        Returns:
            dict with keys: committed, text, word, event
        """
        if ts is None:
            ts = time.time()

        norm = (label or "").strip().lower()
        self._labels.append(norm)

        no_event = {"committed": "", "text": self.text, "word": self.word, "event": "none"}

        # Ignore "nothing"
        if norm == self._nothing_label:
            return no_event

        # Cooldown
        if self._last_commit_ts is not None and (ts - self._last_commit_ts) < self.cooldown_s:
            return no_event

        # Need full window
        if len(self._labels) < self.stable_window:
            return no_event

        # Majority check (excluding "nothing")
        counts = Counter(x for x in self._labels if x != self._nothing_label)
        if not counts:
            return no_event
        top_label, top_count = counts.most_common(1)[0]
        if (top_count / len(self._labels)) < self.majority_ratio:
            return no_event

        # Confidence check
        if confidence < self.min_confidence:
            return no_event

        # Avoid repeating same label
        if self._last_committed_label is not None and top_label == self._last_committed_label:
            return no_event

        result = self._commit(top_label)
        if result.event != "none":
            self._last_commit_ts = ts
            self._last_committed_label = top_label

        return {
            "committed": result.committed,
            "text": self.text,
            "word": self.word,
            "event": result.event,
        }

    def _commit(self, stable_label: str) -> _CommitResult:
        if stable_label == self._space_label:
            if self._text_chars and self._text_chars[-1] == " ":
                return _CommitResult("", "none")
            self._text_chars.append(" ")
            return _CommitResult(" ", "space")

        if stable_label == self._del_label:
            if not self._text_chars:
                return _CommitResult("", "none")
            self._text_chars.pop()
            return _CommitResult("", "delete")

        # Regular letter — uppercase single letters
        committed = stable_label
        if len(committed) == 1 and committed.isalpha():
            committed = committed.upper()
        self._text_chars.append(committed)
        return _CommitResult(committed, "append")
