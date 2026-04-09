"""
word_decoder.py — English word sentence builder.

Accumulates committed words into a running sentence.
"""

from __future__ import annotations

from typing import List


class EnglishWordDecoder:
    """Build a sentence from recognized words.

    Usage::

        decoder = EnglishWordDecoder()
        sentence = decoder.add_word("hello")  # "hello"
        sentence = decoder.add_word("world")  # "hello world"
    """

    def __init__(self) -> None:
        self._words: List[str] = []

    def add_word(self, word: str) -> str:
        """Add a word and return the full sentence so far."""
        word = word.strip()
        if word:
            self._words.append(word)
        return self.get_sentence()

    def get_sentence(self) -> str:
        """Return the accumulated sentence."""
        return " ".join(self._words)

    def clear(self) -> None:
        """Clear the sentence."""
        self._words.clear()

    @property
    def word_count(self) -> int:
        return len(self._words)
