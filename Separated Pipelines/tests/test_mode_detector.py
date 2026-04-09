"""
test_mode_detector.py — Test LETTER / WORD / IDLE switching logic.
"""

import numpy as np
import pytest

from shared.utils.mode_detector import ModeDetector, SignMode


class TestModeDetector:
    def setup_method(self):
        self.det = ModeDetector(
            motion_thresh_letter=0.015,
            motion_thresh_word=0.030,
            smooth_window=3,
        )

    def _make_landmarks(self, offset: float = 0.0) -> np.ndarray:
        """Create a deterministic (63,) landmark array with small offset."""
        return np.full(63, 0.5, dtype=np.float32) + offset

    def test_initial_mode_is_idle(self):
        assert self.det.current_mode == SignMode.IDLE

    def test_stationary_frames_yield_letter_mode(self):
        for _ in range(5):
            mode = self.det.update(self._make_landmarks(0.0))
        assert mode == SignMode.LETTER

    def test_moving_frames_yield_word_mode(self):
        for i in range(10):
            mode = self.det.update(self._make_landmarks(i * 0.1))
        assert mode == SignMode.WORD

    def test_reset(self):
        for i in range(5):
            self.det.update(self._make_landmarks(i * 0.1))
        self.det.reset()
        assert self.det.current_mode == SignMode.IDLE

    def test_hysteresis_prevents_flicker(self):
        """In the hysteresis band, mode should not change."""
        # Get into LETTER mode
        for _ in range(5):
            self.det.update(self._make_landmarks(0.0))
        assert self.det.current_mode == SignMode.LETTER

        # Small motion in hysteresis band — should stay LETTER
        mode = self.det.update(self._make_landmarks(0.02))
        assert mode == SignMode.LETTER
