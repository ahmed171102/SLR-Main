"""
correction_agent.py — GPT-powered confidence-gated correction layer.

Called ONLY when model softmax confidence is below LLM_CONFIDENCE_GATE.
Supports four correction modes:
  1. English letter stream correction (spelling fix)
  2. English word candidate reranking (context-aware)
  3. Arabic letter stream correction
  4. Arabic word candidate reranking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────


@dataclass
class LetterCorrectionRequest:
    raw_text: str
    language: str  # "en" | "ar"


@dataclass
class WordCorrectionRequest:
    candidates: List[Tuple[str, float]]  # [(word, confidence), ...]
    sentence_context: str
    language: str  # "en" | "ar"


@dataclass
class CorrectionResult:
    corrected_text: str
    was_corrected: bool
    used_llm: bool
    latency_ms: float
    cached: bool = False


# ──────────────────────────────────────────────────────
# Correction Agent
# ──────────────────────────────────────────────────────


class LLMCorrectionAgent:
    """Confidence-gated LLM correction agent.

    Usage::

        agent = LLMCorrectionAgent(api_key="sk-...", confidence_gate=0.75)
        result = agent.correct_letters(LetterCorrectionRequest("HEVLO", "en"))
        result = agent.correct_words(WordCorrectionRequest(candidates, ctx, "en"))
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        confidence_gate: float = 0.75,
        timeout_ms: int = 2000,
        cache_size: int = 500,
        temperature: float = 0.2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.confidence_gate = confidence_gate
        self.timeout_ms = timeout_ms
        self.temperature = temperature

        # Simple dict cache (limited size)
        self._cache: dict[str, str] = {}
        self._cache_size = cache_size

    # ── Public API ───────────────────────────────────

    def should_correct(self, confidence: float) -> bool:
        """Return True if confidence is below the gate threshold."""
        return confidence < self.confidence_gate

    def correct_letters(self, request: LetterCorrectionRequest) -> CorrectionResult:
        """Correct a letter stream using LLM if needed."""
        from .prompts import (
            ARABIC_LETTER_CORRECTION_PROMPT,
            ENGLISH_LETTER_CORRECTION_PROMPT,
        )

        if not self.api_key:
            return CorrectionResult(
                corrected_text=request.raw_text,
                was_corrected=False,
                used_llm=False,
                latency_ms=0.0,
            )

        cache_key = f"letter:{request.language}:{request.raw_text}"
        if cache_key in self._cache:
            return CorrectionResult(
                corrected_text=self._cache[cache_key],
                was_corrected=self._cache[cache_key] != request.raw_text,
                used_llm=True,
                latency_ms=0.0,
                cached=True,
            )

        prompt = (
            ENGLISH_LETTER_CORRECTION_PROMPT
            if request.language == "en"
            else ARABIC_LETTER_CORRECTION_PROMPT
        )

        start = time.perf_counter()
        corrected = self._call_openai(prompt, request.raw_text)
        latency = (time.perf_counter() - start) * 1000

        if corrected is None:
            return CorrectionResult(
                corrected_text=request.raw_text,
                was_corrected=False,
                used_llm=True,
                latency_ms=latency,
            )

        self._put_cache(cache_key, corrected)
        return CorrectionResult(
            corrected_text=corrected,
            was_corrected=corrected != request.raw_text,
            used_llm=True,
            latency_ms=latency,
        )

    def correct_words(self, request: WordCorrectionRequest) -> CorrectionResult:
        """Rerank word candidates using LLM context-aware selection."""
        from .prompts import (
            ARABIC_WORD_CORRECTION_PROMPT,
            ENGLISH_WORD_CORRECTION_PROMPT,
        )

        if not self.api_key or not request.candidates:
            top = request.candidates[0][0] if request.candidates else ""
            return CorrectionResult(
                corrected_text=top,
                was_corrected=False,
                used_llm=False,
                latency_ms=0.0,
            )

        candidates_str = ", ".join(
            f"{w} ({c:.2f})" for w, c in request.candidates
        )
        user_msg = (
            f"Sentence so far: \"{request.sentence_context}\"\n"
            f"Candidates: {candidates_str}\n"
            f"Which word fits best?"
        )

        cache_key = f"word:{request.language}:{user_msg}"
        if cache_key in self._cache:
            return CorrectionResult(
                corrected_text=self._cache[cache_key],
                was_corrected=True,
                used_llm=True,
                latency_ms=0.0,
                cached=True,
            )

        prompt = (
            ENGLISH_WORD_CORRECTION_PROMPT
            if request.language == "en"
            else ARABIC_WORD_CORRECTION_PROMPT
        )

        start = time.perf_counter()
        selected = self._call_openai(prompt, user_msg)
        latency = (time.perf_counter() - start) * 1000

        top_word = request.candidates[0][0]
        if selected is None:
            return CorrectionResult(
                corrected_text=top_word,
                was_corrected=False,
                used_llm=True,
                latency_ms=latency,
            )

        self._put_cache(cache_key, selected)
        return CorrectionResult(
            corrected_text=selected,
            was_corrected=selected != top_word,
            used_llm=True,
            latency_ms=latency,
        )

    # ── Private ──────────────────────────────────────

    def _put_cache(self, key: str, value: str) -> None:
        if len(self._cache) >= self._cache_size:
            # Evict oldest (first) entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def _call_openai(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Call OpenAI chat completion. Returns response text or None on error."""
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=100,
                timeout=self.timeout_ms / 1000,
            )
            text = response.choices[0].message.content.strip()
            return text
        except ImportError:
            logger.warning("openai package not installed — LLM correction disabled")
            return None
        except Exception as e:
            logger.warning("OpenAI call failed: %s", e)
            return None
