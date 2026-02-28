"""
pipeline.py — Arabic ArSL real-time inference pipeline.

Coordinates:  MediaPipe → mode detection → letter/word prediction
              → decoder → LLM correction → UI-ready dict.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from config.settings import (
    AR_LETTER_CONFIDENCE_THRESHOLD,
    AR_WORD_CONFIDENCE_THRESHOLD,
    LLM_CONFIDENCE_GATE,
)
from shared.utils.mediapipe_extractor import MediaPipeExtractor
from shared.utils.mode_detector import ModeDetector, SignMode
from arabic_pipeline.models.letter_predictor import ArabicLetterPredictor
from arabic_pipeline.models.word_predictor import ArabicWordPredictor
from arabic_pipeline.decoders.letter_decoder import ArabicLetterDecoder
from arabic_pipeline.decoders.word_decoder import ArabicWordDecoder
from llm_agent.correction_agent import LLMCorrectionAgent

logger = logging.getLogger(__name__)


class ArabicPipeline:
    """Full Arabic ArSL inference pipeline — one frame in, result dict out."""

    def __init__(
        self,
        enable_llm: bool = True,
        openai_api_key: Optional[str] = None,
    ) -> None:
        # ── sub-components ──
        self.extractor = MediaPipeExtractor()
        self.mode_detector = ModeDetector()

        self.letter_predictor = ArabicLetterPredictor()
        self.word_predictor = ArabicWordPredictor()

        self.letter_decoder = ArabicLetterDecoder()
        self.word_decoder = ArabicWordDecoder()

        # ── LLM ──
        self._enable_llm = enable_llm
        self._llm: Optional[LLMCorrectionAgent] = None
        if enable_llm:
            try:
                self._llm = LLMCorrectionAgent(
                    language="ar",
                    api_key=openai_api_key,
                )
            except Exception as exc:
                logger.warning("LLM agent disabled (AR): %s", exc)
                self._enable_llm = False

        # ── word-mode frame buffer ──
        self._frame_buffer: list[np.ndarray] = []
        self._WORD_SEQ_LEN = 30

        # ── stats ──
        self._frames_processed = 0

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────
    def process_frame(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """Run one BGR frame through the full Arabic pipeline."""
        self._frames_processed += 1

        result: Dict[str, Any] = {
            "language": "ar",
            "mode": "idle",
            "letter_text": self.letter_decoder.text,
            "word_sentence": self.word_decoder.sentence,
            "last_prediction": "",
            "arabic_char": "",
            "confidence": 0.0,
            "motion": 0.0,
            "llm_corrected": False,
        }

        # 1 — hand landmarks
        kp = self.extractor.extract(bgr_frame)
        if kp is None:
            self.mode_detector.update(None)
            return result

        # 2 — mode detection
        mode = self.mode_detector.update(kp)
        result["motion"] = float(self.mode_detector.current_motion)
        result["mode"] = mode.value

        # 3 — branch by mode
        if mode == SignMode.LETTER:
            self._frame_buffer.clear()
            result = self._handle_letter(kp, result)
        elif mode == SignMode.WORD:
            result = self._handle_word(kp, result)

        return result

    def reset(self) -> None:
        """Clear all state."""
        self.letter_decoder.reset()
        self.word_decoder.reset()
        self.mode_detector.reset()
        self._frame_buffer.clear()

    # ──────────────────────────────────────────────
    # Letter branch
    # ──────────────────────────────────────────────
    def _handle_letter(
        self, kp: np.ndarray, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        romanized, arabic_char, conf = self.letter_predictor.predict(kp)
        result["last_prediction"] = romanized
        result["arabic_char"] = arabic_char
        result["confidence"] = float(conf)

        if conf >= AR_LETTER_CONFIDENCE_THRESHOLD:
            dec_out = self.letter_decoder.update(romanized, arabic_char, conf)
            result["letter_text"] = dec_out["text"]

            # LLM correction when a word is complete
            if dec_out["event"] == "space_committed" and self._llm:
                self._try_llm_letter_correction(result)

        return result

    def _try_llm_letter_correction(self, result: Dict[str, Any]) -> None:
        word = self.letter_decoder.get_current_word_for_correction()
        if not word:
            return
        try:
            cr = self._llm.correct_letters(
                list(word),
                [0.8] * len(word),  # average confidence placeholder
            )
            if cr.corrected:
                self.letter_decoder.apply_correction(cr.text)
                result["letter_text"] = self.letter_decoder.text
                result["llm_corrected"] = True
        except Exception as exc:
            logger.debug("LLM letter correction (AR) failed: %s", exc)

    # ──────────────────────────────────────────────
    # Word branch
    # ──────────────────────────────────────────────
    def _handle_word(
        self, kp: np.ndarray, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._frame_buffer.append(kp)
        if len(self._frame_buffer) < self._WORD_SEQ_LEN:
            return result

        seq = np.array(self._frame_buffer[-self._WORD_SEQ_LEN :])
        self._frame_buffer = self._frame_buffer[-self._WORD_SEQ_LEN :]

        word, word_id, conf = self.word_predictor.predict(seq)
        result["last_prediction"] = word
        result["confidence"] = float(conf)

        if conf >= AR_WORD_CONFIDENCE_THRESHOLD:
            dec_out = self.word_decoder.update(word, word_id, conf)
            result["word_sentence"] = dec_out["sentence"]

            # LLM correction for low-mid confidence words
            if dec_out["event"] == "word_added" and self._llm:
                self._try_llm_word_correction(word, conf, result)

        return result

    def _try_llm_word_correction(
        self, word: str, conf: float, result: Dict[str, Any]
    ) -> None:
        if conf >= LLM_CONFIDENCE_GATE:
            return
        try:
            cr = self._llm.correct_word(word, conf)
            if cr.corrected:
                self.word_decoder.replace_last_word(cr.text)
                result["word_sentence"] = self.word_decoder.sentence
                result["llm_corrected"] = True
        except Exception as exc:
            logger.debug("LLM word correction (AR) failed: %s", exc)
