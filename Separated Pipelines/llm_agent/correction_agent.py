"""
correction_agent.py — Confidence-gated LLM correction layer.

This agent sits between the raw model predictions and the final output.
It is called ONLY when the model's maximum softmax confidence falls below
a configurable threshold, saving API calls and cost.

Supports:
  - English letter-stream correction (spelling)
  - English word-candidate reranking (context)
  - Arabic letter-stream correction (spelling)
  - Arabic word-candidate reranking (context)

Uses GPT-4o-mini by default (~$0.15/1M tokens — cheapest reliable option).
Falls back to raw prediction on API errors or timeouts.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.settings import (
    LLM_API_KEY,
    LLM_CACHE_SIZE,
    LLM_CONFIDENCE_GATE,
    LLM_ENABLED,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_MS,
)
from llm_agent.prompts import (
    ARABIC_LETTER_CORRECTION_PROMPT,
    ARABIC_WORD_CORRECTION_PROMPT,
    ENGLISH_LETTER_CORRECTION_PROMPT,
    ENGLISH_WORD_CORRECTION_PROMPT,
)

logger = logging.getLogger(__name__)


# ─── Lightweight LRU cache ───────────────────────────────────

class _LRUCache:
    """Simple OrderedDict-based LRU cache (no external deps)."""

    def __init__(self, max_size: int = 500):
        self._max = max_size
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self._max:
            self._store.popitem(last=False)


# ─── Data classes ────────────────────────────────────────────

@dataclass
class LetterCorrectionRequest:
    """A stream of predicted letters with their confidences."""
    letters: List[str]                 # e.g. ["H", "E", "L", "O"]
    confidences: List[float]           # e.g. [0.9, 0.85, 0.7, 0.6]
    language: str = "en"               # "en" or "ar"


@dataclass
class WordCorrectionRequest:
    """A set of candidate word predictions to rerank."""
    candidates: List[Tuple[str, float]]  # [(word, confidence), ...]
    sentence_context: str = ""           # sentence built so far
    language: str = "en"                 # "en" or "ar"


@dataclass
class CorrectionResult:
    """Result of an LLM correction call."""
    corrected_text: str
    was_corrected: bool          # True if LLM changed the prediction
    used_llm: bool               # True if LLM was actually called
    latency_ms: float = 0.0
    cached: bool = False


# ─── Main Agent ──────────────────────────────────────────────

class LLMCorrectionAgent:
    """Confidence-gated LLM correction agent.

    Usage:
        agent = LLMCorrectionAgent()

        # Letter correction
        result = agent.correct_letters(
            LetterCorrectionRequest(letters=["H","E","L","O"], confidences=[0.9,0.85,0.7,0.6])
        )

        # Word correction
        result = agent.correct_word(
            WordCorrectionRequest(candidates=[("help", 0.6), ("hello", 0.55)], sentence_context="my name")
        )
    """

    def __init__(
        self,
        enabled: bool = LLM_ENABLED,
        confidence_gate: float = LLM_CONFIDENCE_GATE,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
        provider: str = LLM_PROVIDER,
        timeout_ms: float = LLM_TIMEOUT_MS,
        temperature: float = LLM_TEMPERATURE,
        cache_size: int = LLM_CACHE_SIZE,
    ):
        self.enabled = enabled
        self.confidence_gate = confidence_gate
        self.model = model
        self.provider = provider
        self.timeout_ms = timeout_ms
        self.temperature = temperature
        self._cache = _LRUCache(cache_size)
        self._client = None

        # Stats
        self.total_calls = 0
        self.llm_calls = 0
        self.cache_hits = 0
        self.fallback_count = 0

        # Initialize OpenAI client if enabled
        if self.enabled and api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                logger.info("LLM Agent initialized with %s (%s)", provider, model)
            except ImportError:
                logger.warning(
                    "openai package not installed. LLM correction disabled. "
                    "Install with: pip install openai"
                )
                self.enabled = False
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s", e)
                self.enabled = False
        elif self.enabled and not api_key:
            logger.warning(
                "LLM_ENABLED=True but OPENAI_API_KEY is empty. "
                "Set env var OPENAI_API_KEY or disable in settings."
            )
            self.enabled = False

    # ── Letter correction ────────────────────────────────────

    def correct_letters(self, request: LetterCorrectionRequest) -> CorrectionResult:
        """Correct a letter sequence if confidence is below the gate.

        The gate checks the MINIMUM confidence in the sequence — if any letter
        is uncertain, the whole sequence gets sent to the LLM.
        """
        self.total_calls += 1

        raw_text = "".join(request.letters)

        # Gate check: skip LLM if all confidences are high
        if not self.enabled or not request.confidences:
            return CorrectionResult(
                corrected_text=raw_text, was_corrected=False, used_llm=False
            )

        min_conf = min(request.confidences) if request.confidences else 1.0
        if min_conf >= self.confidence_gate:
            return CorrectionResult(
                corrected_text=raw_text, was_corrected=False, used_llm=False
            )

        # Check cache
        cache_key = self._make_cache_key("letter", request.language, raw_text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return CorrectionResult(
                corrected_text=cached,
                was_corrected=(cached != raw_text),
                used_llm=False,
                cached=True,
            )

        # Build the user message
        pairs = [
            f"  {l} (conf: {c:.2f})"
            for l, c in zip(request.letters, request.confidences)
        ]
        user_msg = "Letter sequence:\n" + "\n".join(pairs)

        system_prompt = (
            ENGLISH_LETTER_CORRECTION_PROMPT
            if request.language == "en"
            else ARABIC_LETTER_CORRECTION_PROMPT
        )

        corrected = self._call_llm(system_prompt, user_msg)
        if corrected is None:
            self.fallback_count += 1
            return CorrectionResult(
                corrected_text=raw_text, was_corrected=False, used_llm=True
            )

        self._cache.put(cache_key, corrected)
        return CorrectionResult(
            corrected_text=corrected,
            was_corrected=(corrected != raw_text),
            used_llm=True,
        )

    # ── Word correction ──────────────────────────────────────

    def correct_word(self, request: WordCorrectionRequest) -> CorrectionResult:
        """Rerank / correct word candidates if top confidence is below the gate."""
        self.total_calls += 1

        if not request.candidates:
            return CorrectionResult(
                corrected_text="", was_corrected=False, used_llm=False
            )

        top_word, top_conf = request.candidates[0]

        # Gate check
        if not self.enabled or top_conf >= self.confidence_gate:
            return CorrectionResult(
                corrected_text=top_word, was_corrected=False, used_llm=False
            )

        # Check cache
        cand_str = "|".join(f"{w}:{c:.3f}" for w, c in request.candidates)
        cache_key = self._make_cache_key(
            "word", request.language, f"{cand_str}||{request.sentence_context}"
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return CorrectionResult(
                corrected_text=cached,
                was_corrected=(cached != top_word),
                used_llm=False,
                cached=True,
            )

        # Build user message
        cand_lines = [
            f"  {w} (conf: {c:.3f})" for w, c in request.candidates
        ]
        user_msg = (
            f"Sentence so far: \"{request.sentence_context}\"\n"
            f"Word candidates:\n" + "\n".join(cand_lines)
        )

        system_prompt = (
            ENGLISH_WORD_CORRECTION_PROMPT
            if request.language == "en"
            else ARABIC_WORD_CORRECTION_PROMPT
        )

        corrected = self._call_llm(system_prompt, user_msg)
        if corrected is None:
            self.fallback_count += 1
            return CorrectionResult(
                corrected_text=top_word, was_corrected=False, used_llm=True
            )

        self._cache.put(cache_key, corrected)
        return CorrectionResult(
            corrected_text=corrected,
            was_corrected=(corrected != top_word),
            used_llm=True,
        )

    # ── Stats ────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "fallbacks": self.fallback_count,
        }

    # ── Private helpers ──────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Call the LLM API with timeout and error handling.

        Returns the corrected text or None on failure.
        """
        if self._client is None:
            return None

        self.llm_calls += 1
        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=200,
                timeout=self.timeout_ms / 1000.0,
            )
            result = response.choices[0].message.content.strip()
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug("LLM response in %.0f ms: %s", elapsed, result)
            return result

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("LLM call failed (%.0f ms): %s", elapsed, e)
            return None

    @staticmethod
    def _make_cache_key(task: str, lang: str, content: str) -> str:
        """Create a deterministic cache key."""
        raw = f"{task}:{lang}:{content}"
        return hashlib.md5(raw.encode()).hexdigest()


# ── Async wrapper (for FastAPI) ──────────────────────────────

class AsyncLLMCorrectionAgent(LLMCorrectionAgent):
    """Async version that wraps sync calls for use in FastAPI routes.

    The OpenAI SDK's sync client is used under the hood with
    asyncio.to_thread() to avoid blocking the event loop.
    """

    async def acorrect_letters(
        self, request: LetterCorrectionRequest
    ) -> CorrectionResult:
        return await asyncio.to_thread(self.correct_letters, request)

    async def acorrect_word(
        self, request: WordCorrectionRequest
    ) -> CorrectionResult:
        return await asyncio.to_thread(self.correct_word, request)
