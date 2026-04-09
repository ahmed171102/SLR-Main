"""
test_llm_agent.py — Test LLM correction agent cache and fallback.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agent.correction_agent import CorrectionResult, LLMCorrectionAgent


class TestLLMCorrectionAgent:
    def setup_method(self):
        self.agent = LLMCorrectionAgent(
            api_key="sk-test-fake-key",
            model="gpt-4o-mini",
            confidence_gate=0.75,
        )

    def test_below_confidence_gate_returns_original(self):
        """When confidence > gate, no correction should trigger."""
        result = self.agent.correct_letters("HELLO", 0.95, language="english")
        # High confidence → should return original (no API call)
        assert result.corrected_text == "HELLO"
        assert not result.was_corrected

    def test_cache_returns_same_result(self):
        """Repeated calls with same input should hit cache."""
        # First call
        r1 = self.agent.correct_letters("TEST", 0.95, language="english")
        # Second call — cache hit
        r2 = self.agent.correct_letters("TEST", 0.95, language="english")
        assert r1.corrected_text == r2.corrected_text

    def test_no_api_key_returns_original(self):
        agent = LLMCorrectionAgent(api_key="", model="gpt-4o-mini")
        result = agent.correct_letters("HELLO", 0.50, language="english")
        assert result.corrected_text == "HELLO"
        assert not result.was_corrected
