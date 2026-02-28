"""
pipeline.py — Complete English (ASL) inference pipeline.

This is the main entry point for the English sign-language screen.
It orchestrates:
  1. MediaPipe landmark extraction
  2. Motion-based mode detection (letter vs word)
  3. Letter MLP prediction → letter decoder → text
  4. Word BiLSTM prediction → word decoder → sentence
  5. LLM correction agent (confidence-gated)

Usage (webcam loop):
    from english_pipeline import EnglishPipeline

    pipeline = EnglishPipeline()
    pipeline.load()

    while True:
        frame = webcam.read()
        result = pipeline.process_frame(frame)
        print(result["text"], result["mode"])
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict, Optional

import cv2
import numpy as np

from config.settings import (
    EN_LETTER_CONFIDENCE_THRESHOLD,
    EN_WORD_CONFIDENCE_THRESHOLD,
    LLM_CONFIDENCE_GATE,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
)
from shared.utils.mediapipe_extractor import MediaPipeExtractor
from shared.utils.mode_detector import ModeDetector, SignMode
from english_pipeline.models.letter_predictor import EnglishLetterPredictor
from english_pipeline.models.word_predictor import EnglishWordPredictor
from english_pipeline.decoders.letter_decoder import EnglishLetterDecoder
from english_pipeline.decoders.word_decoder import EnglishWordDecoder
from llm_agent.correction_agent import (
    LLMCorrectionAgent,
    LetterCorrectionRequest,
    WordCorrectionRequest,
)

logger = logging.getLogger(__name__)


class EnglishPipeline:
    """Complete English (ASL) sign language recognition pipeline.

    Fully self-contained: has its own MediaPipe instance, models,
    decoders, mode detector, frame buffer, and LLM agent.
    Does NOT share any state with the Arabic pipeline.
    """

    def __init__(self, llm_agent: Optional[LLMCorrectionAgent] = None) -> None:
        # Components (lazy-loaded)
        self._extractor: Optional[MediaPipeExtractor] = None
        self._mode_detector: Optional[ModeDetector] = None
        self._letter_predictor: Optional[EnglishLetterPredictor] = None
        self._word_predictor: Optional[EnglishWordPredictor] = None
        self._letter_decoder: Optional[EnglishLetterDecoder] = None
        self._word_decoder: Optional[EnglishWordDecoder] = None
        self._llm_agent = llm_agent

        # Word frame buffer
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

        # State
        self._loaded = False
        self._frame_count = 0

    # ── Loading ──────────────────────────────────────────────

    def load(self) -> None:
        """Load all models and initialize components. Call once before processing."""
        if self._loaded:
            return

        logger.info("Loading English (ASL) pipeline...")
        t0 = time.perf_counter()

        self._extractor = MediaPipeExtractor()
        self._mode_detector = ModeDetector()
        self._letter_predictor = EnglishLetterPredictor()
        self._letter_predictor.load()
        self._word_predictor = EnglishWordPredictor()
        self._word_predictor.load()
        self._letter_decoder = EnglishLetterDecoder()
        self._word_decoder = EnglishWordDecoder()

        if self._llm_agent is None:
            self._llm_agent = LLMCorrectionAgent()

        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info("English pipeline loaded in %.1f s", elapsed)

    # ── Main processing ──────────────────────────────────────

    def process_frame(self, bgr_frame: np.ndarray) -> Dict:
        """Process a single BGR webcam frame through the full English pipeline.

        Parameters
        ----------
        bgr_frame : np.ndarray
            HxWx3 uint8 BGR image (as returned by cv2.VideoCapture.read()).

        Returns
        -------
        dict with keys:
            mode: "letter" | "word" | "idle"
            letter_text: accumulated letter text
            word_sentence: accumulated word sentence
            last_prediction: latest label/word
            confidence: latest confidence
            motion: smoothed motion value
            llm_corrected: bool — whether LLM modified the prediction
        """
        if not self._loaded:
            self.load()

        self._frame_count += 1
        ts = time.time()

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        landmarks = self._extractor.extract(rgb)

        # Mode detection
        mode = self._mode_detector.update(landmarks)
        motion = self._mode_detector.last_motion

        result = {
            "mode": mode.value,
            "letter_text": self._letter_decoder.text,
            "word_sentence": self._word_decoder.sentence,
            "last_prediction": "",
            "confidence": 0.0,
            "motion": motion,
            "llm_corrected": False,
            "frame": self._frame_count,
        }

        if landmarks is None:
            return result

        # Always buffer frames for word model
        self._frame_buffer.append(landmarks)

        # ── LETTER MODE ──────────────────────────────────────
        if mode == SignMode.LETTER:
            label, conf = self._letter_predictor.predict(landmarks)
            event = self._letter_decoder.update(label, conf, ts)

            result["last_prediction"] = label
            result["confidence"] = conf
            result["letter_text"] = event.text

            # LLM correction on completed words (when space is committed)
            if event.event == "space" and self._llm_agent:
                result["llm_corrected"] = self._try_letter_correction()
                result["letter_text"] = self._letter_decoder.text

        # ── WORD MODE ────────────────────────────────────────
        elif mode == SignMode.WORD and len(self._frame_buffer) == SEQUENCE_LENGTH:
            sequence = np.array(self._frame_buffer, dtype=np.float32)
            word, word_id, conf = self._word_predictor.predict(sequence)

            result["last_prediction"] = word
            result["confidence"] = conf

            if conf >= EN_WORD_CONFIDENCE_THRESHOLD:
                # Try LLM correction on low-confidence words
                corrected_word = word
                if self._llm_agent and conf < LLM_CONFIDENCE_GATE:
                    corrected_word, was_corrected = self._try_word_correction(
                        sequence, word, conf
                    )
                    result["llm_corrected"] = was_corrected

                commit = self._word_decoder.update(corrected_word, word_id, conf, ts)
                result["word_sentence"] = commit["sentence"]

                if commit["event"] == "word_added":
                    self._frame_buffer.clear()

        return result

    # ── LLM correction helpers ───────────────────────────────

    def _try_letter_correction(self) -> bool:
        """Attempt LLM correction on the most recently completed word."""
        letters, confs = self._letter_decoder.get_current_word_for_correction()
        if not letters or len(letters) < 2:
            return False

        # Only correct if any letter has low confidence
        if min(confs) >= LLM_CONFIDENCE_GATE:
            return False

        req = LetterCorrectionRequest(
            letters=letters, confidences=confs, language="en"
        )
        result = self._llm_agent.correct_letters(req)

        if result.was_corrected:
            self._letter_decoder.apply_correction(result.corrected_text)
            logger.info(
                "LLM corrected letters: %s → %s",
                "".join(letters),
                result.corrected_text,
            )
            return True
        return False

    def _try_word_correction(
        self, sequence: np.ndarray, top_word: str, top_conf: float
    ) -> tuple[str, bool]:
        """Attempt LLM reranking of word candidates."""
        top_k = self._word_predictor.predict_top_k(sequence, k=5)
        candidates = [(w, c) for w, _, c in top_k]

        req = WordCorrectionRequest(
            candidates=candidates,
            sentence_context=self._word_decoder.sentence,
            language="en",
        )
        result = self._llm_agent.correct_word(req)
        return result.corrected_text, result.was_corrected

    # ── State management ─────────────────────────────────────

    def reset(self) -> None:
        """Reset all decoders and buffers (not models)."""
        if self._letter_decoder:
            self._letter_decoder.reset()
        if self._word_decoder:
            self._word_decoder.reset()
        if self._mode_detector:
            self._mode_detector.reset()
        self._frame_buffer.clear()
        self._frame_count = 0

    @property
    def letter_text(self) -> str:
        return self._letter_decoder.text if self._letter_decoder else ""

    @property
    def word_sentence(self) -> str:
        return self._word_decoder.sentence if self._word_decoder else ""

    @property
    def full_output(self) -> str:
        """Combined letter text + word sentence."""
        parts = []
        if self._word_decoder and self._word_decoder.sentence:
            parts.append(self._word_decoder.sentence)
        if self._letter_decoder and self._letter_decoder.text:
            parts.append(self._letter_decoder.text)
        return " ".join(parts)

    def get_stats(self) -> Dict:
        """Return pipeline statistics."""
        stats = {
            "frames_processed": self._frame_count,
            "letter_text": self.letter_text,
            "word_sentence": self.word_sentence,
        }
        if self._llm_agent:
            stats["llm_stats"] = self._llm_agent.get_stats()
        return stats

    def close(self) -> None:
        """Release resources."""
        if self._extractor:
            self._extractor.close()
