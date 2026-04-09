"""
word_decoder.py — Arabic word sentence builder.

Same concept as English but handles RTL text via arabic_reshaper.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_RTL = True
except ImportError:
    _HAS_RTL = False


class ArabicWordDecoder:
    """Build an Arabic sentence from recognized words.

    Usage::

        decoder = ArabicWordDecoder()
        sentence = decoder.add_word("مرحبا")
        sentence = decoder.add_word("عالم")
    """

    def __init__(self) -> None:
        self._words: List[str] = []

    def add_word(self, word: str) -> str:
        word = word.strip()
        if word:
            self._words.append(word)
        return self.get_sentence()

    def get_sentence(self) -> str:
        return " ".join(self._words)

    def get_sentence_rtl(self) -> str:
        """Return sentence formatted for RTL display."""
        text = self.get_sentence()
        if not _HAS_RTL or not text:
            return text
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    def clear(self) -> None:
        self._words.clear()

    @property
    def word_count(self) -> int:
        return len(self._words)
