"""letter_stream_decoder.py

A tiny, dependency-free utility that turns per-frame predictions
(label + confidence) into a text string.

Designed for webcam pipelines that emit one prediction per frame.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class _CommitResult:
    committed: str
    event: str  # "append"|"space"|"delete"|"none"


class LetterStreamDecoder:
    """Decode a stream of per-frame predictions into text.

    Core idea:
    - Maintain a window (deque) of the last N labels.
    - When a label becomes stable (majority in the window) and the current
      confidence is high enough, commit it (append letter / space / delete).
    - Use a cooldown to avoid repeated commits across adjacent frames.

    Control labels:
    - "nothing": ignored (no event; still contributes to stability window)
    - "space": inserts a single space (avoids consecutive spaces)
    - "del": deletes last character
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        stable_window: int = 5,
        majority_ratio: float = 0.7,
        cooldown_s: float = 0.6,
        control_labels: Tuple[str, str, str] = ("space", "del", "nothing"),
    ) -> None:
        if stable_window <= 0:
            raise ValueError("stable_window must be > 0")
        if not (0.0 < majority_ratio <= 1.0):
            raise ValueError("majority_ratio must be in (0, 1]")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")

        self.min_confidence = float(min_confidence)
        self.stable_window = int(stable_window)
        self.majority_ratio = float(majority_ratio)
        self.cooldown_s = float(cooldown_s)

        self._control_labels = tuple(str(x).strip().lower() for x in control_labels)
        self._nothing_label = self._control_labels[2] if len(self._control_labels) >= 3 else "nothing"
        self._space_label = self._control_labels[0] if len(self._control_labels) >= 1 else "space"
        self._del_label = self._control_labels[1] if len(self._control_labels) >= 2 else "del"

        self._labels: Deque[str] = deque(maxlen=self.stable_window)
        self._text_chars: list[str] = []

        self._last_commit_ts: Optional[float] = None
        self._last_committed_label: Optional[str] = None

    def reset(self) -> None:
        """Reset decoder state (window, text, cooldown, last committed label)."""
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
        """Update decoder with a single frame prediction.

        Args:
            label: predicted label for the current frame.
            confidence: confidence score for the current frame.
            ts: timestamp in seconds (time.time() style). If None, uses time.time().

        Returns:
            dict with keys:
              - committed: str ("" if nothing committed)
              - text: full accumulated text
              - word: current word (after last space)
              - event: "append"|"space"|"delete"|"none"
        """
        if ts is None:
            ts = time.time()

        raw_label = "" if label is None else str(label).strip()
        norm_label = raw_label.lower()
        conf = float(confidence)

        # Always push into the stability window.
        self._labels.append(norm_label)

        # Rule: ignore "nothing".
        if norm_label == self._nothing_label:
            return {
                "committed": "",
                "text": self.text,
                "word": self.word,
                "event": "none",
            }

        # Rule: cooldown after any commit.
        if self._last_commit_ts is not None and (ts - self._last_commit_ts) < self.cooldown_s:
            return {
                "committed": "",
                "text": self.text,
                "word": self.word,
                "event": "none",
            }

        # Need a full window to decide stability.
        if len(self._labels) < self.stable_window:
            return {
                "committed": "",
                "text": self.text,
                "word": self.word,
                "event": "none",
            }

        # Rule: stable majority label in window (excluding "nothing").
        counts = Counter(x for x in self._labels if x != self._nothing_label)
        if not counts:
            return {
                "committed": "",
                "text": self.text,
                "word": self.word,
                "event": "none",
            }

        top_label, top_count = counts.most_common(1)[0]
        if (top_count / len(self._labels)) < self.majority_ratio:
            return {
                "committed": "",
                "text": self.text,
                "word": self.word,
                "event": "none",
            }

        # Rule: confidence threshold.
        if conf < self.min_confidence:
            return {
                "committed": "",
                "text": self.text,
                "word": self.word,
                "event": "none",
            }

        # Rule: avoid committing same label repeatedly.
        if self._last_committed_label is not None and top_label == self._last_committed_label:
            return {
                "committed": "",
                "text": self.text,
                "word": self.word,
                "event": "none",
            }

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
        # Control: space
        if stable_label == self._space_label:
            if self._text_chars and self._text_chars[-1] == " ":
                return _CommitResult(committed="", event="none")
            self._text_chars.append(" ")
            return _CommitResult(committed=" ", event="space")

        # Control: delete
        if stable_label == self._del_label:
            if not self._text_chars:
                return _CommitResult(committed="", event="none")
            self._text_chars.pop()
            return _CommitResult(committed="", event="delete")

        # Letter / token
        committed = stable_label
        if len(committed) == 1 and committed.isalpha():
            committed = committed.upper()
        self._text_chars.append(committed)
        return _CommitResult(committed=committed, event="append")


def _print_usage() -> None:
    sys.stderr.write(
        "\n".join(
            [
                "LetterStreamDecoder demo (reads from stdin)",
                "", 
                "Input formats (one per line):",
                "  <label>",
                "  <label> <confidence>",
                "  <label> <confidence> <timestamp_seconds>",
                "", 
                "Examples:",
                "  a 0.92",
                "  space 0.95",
                "  del 0.99",
                "", 
                "Tip: Ctrl+Z then Enter (Windows) to end input.",
                "",
            ]
        )
        + "\n"
    )


def _parse_line(line: str) -> Optional[Tuple[str, float, Optional[float]]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) == 1:
        return parts[0], 1.0, None
    if len(parts) == 2:
        return parts[0], float(parts[1]), None
    return parts[0], float(parts[1]), float(parts[2])


if __name__ == "__main__":
    _print_usage()
    decoder = LetterStreamDecoder()

    for raw in sys.stdin:
        parsed = _parse_line(raw)
        if parsed is None:
            continue
        lbl, conf, ts = parsed
        out = decoder.update(lbl, conf, ts)
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
