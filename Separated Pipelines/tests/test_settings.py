"""
test_settings.py — Verify that config paths resolve correctly.
"""

import os
import pytest

from config.settings import Settings


class TestSettings:
    """Smoke-test that all model path attributes exist and are strings."""

    def test_asl_letter_mlp_path_is_str(self):
        assert isinstance(Settings.ASL_LETTER_MLP_PATH, str)

    def test_arsl_letter_mlp_path_is_str(self):
        assert isinstance(Settings.ARSL_LETTER_MLP_PATH, str)

    def test_asl_word_lstm_path_is_str(self):
        assert isinstance(Settings.ASL_WORD_LSTM_PATH, str)

    def test_mediapipe_defaults(self):
        assert Settings.MP_MAX_HANDS >= 1
        assert 0.0 < Settings.MP_DETECTION_CONF <= 1.0
        assert 0.0 < Settings.MP_TRACKING_CONF <= 1.0

    def test_thresholds_within_range(self):
        assert 0.0 < Settings.ENGLISH_LETTER_CONFIDENCE_THRESHOLD <= 1.0
        assert 0.0 < Settings.ARABIC_LETTER_CONFIDENCE_THRESHOLD <= 1.0

    def test_fastapi_port(self):
        assert isinstance(Settings.FASTAPI_PORT, int)
        assert 1024 <= Settings.FASTAPI_PORT <= 65535

    @pytest.mark.skipif(
        not os.path.exists(Settings.ASL_LETTER_MLP_PATH),
        reason="Model file not present on this machine",
    )
    def test_asl_letter_model_file_exists(self):
        assert os.path.isfile(Settings.ASL_LETTER_MLP_PATH)
