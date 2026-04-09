"""
test_letter_decoder.py — Test English + Arabic letter decoders.
"""

import time
import pytest

from english_pipeline.decoders.letter_decoder import EnglishLetterDecoder
from arabic_pipeline.decoders.letter_decoder import ArabicLetterDecoder


class TestEnglishLetterDecoder:
    def setup_method(self):
        self.dec = EnglishLetterDecoder(
            min_confidence=0.80,
            stable_window=3,
            majority_ratio=0.60,
            cooldown_s=0.0,  # disable cooldown for fast tests
        )

    def _feed(self, label: str, conf: float = 0.95, n: int = 3):
        result = None
        t = time.time()
        for i in range(n):
            result = self.dec.update(label, conf, ts=t + i * 0.01)
        return result

    def test_letter_commit(self):
        r = self._feed("A")
        assert r["text"] == "A"
        assert r["event"] == "append"

    def test_nothing_label_ignored(self):
        r = self._feed("nothing")
        assert r["text"] == ""
        assert r["event"] == "none"

    def test_space_commit(self):
        self._feed("A")
        r = self._feed("space")
        assert r["text"] == "A "
        assert r["event"] == "space"

    def test_delete(self):
        self._feed("A")
        r = self._feed("del")
        assert r["text"] == ""
        assert r["event"] == "delete"

    def test_no_duplicate_commit(self):
        self._feed("B")
        r = self._feed("B")
        assert r["text"] == "B"  # not "BB"

    def test_reset_clears_text(self):
        self._feed("X")
        self.dec.reset()
        assert self.dec.text == ""


class TestArabicLetterDecoder:
    def setup_method(self):
        self.dec = ArabicLetterDecoder(
            min_confidence=0.85,
            stable_window=3,
            majority_ratio=0.60,
            cooldown_s=0.0,
        )

    def _feed(self, label: str, conf: float = 0.95, n: int = 3):
        result = None
        t = time.time()
        for i in range(n):
            result = self.dec.update(label, conf, ts=t + i * 0.01)
        return result

    def test_arabic_letter_commit(self):
        r = self._feed("alef")
        assert r["text"] == "alef"
        assert r["event"] == "append"

    def test_below_threshold_no_commit(self):
        r = self._feed("ba", conf=0.50)
        assert r["text"] == ""
