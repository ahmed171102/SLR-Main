"""
coordinator.py — English (ASL) inference coordinator.

Orchestrates the full pipeline:
  Camera frame → MediaPipe → ModeDetector → Predictor → Decoder → LLM correction
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...config import settings
from ...shared.models.model_loader import (
    load_keras_model,
    load_label_encoder_from_csv,
    load_shared_vocabulary,
    load_word_classes,
)
from ...shared.utils.mediapipe_extractor import MediaPipeExtractor
from ...shared.utils.mode_detector import ModeDetector, SignMode
from ...llm_agent.correction_agent import (
    CorrectionResult,
    LetterCorrectionRequest,
    LLMCorrectionAgent,
    WordCorrectionRequest,
)
from ..decoders.letter_decoder import EnglishLetterDecoder
from ..decoders.word_decoder import EnglishWordDecoder
from ..models.letter_predictor import EnglishLetterPredictor
from ..models.word_predictor import EnglishWordPredictor

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of processing a single frame."""
    mode: str              # "letter" | "word" | "idle"
    prediction: str        # current predicted label/word
    confidence: float      # model confidence
    committed: str         # newly committed text (may be "")
    sentence: str          # full accumulated sentence
    event: str             # "append" | "space" | "delete" | "none"
    llm_corrected: bool    # whether LLM correction was applied


class EnglishPipelineCoordinator:
    """Full English (ASL) inference pipeline.

    Usage::

        coordinator = EnglishPipelineCoordinator()
        coordinator.load_models()

        # In camera loop:
        rgb_frame = ...  # H×W×3 uint8
        result = coordinator.process_frame(rgb_frame)
        print(result.sentence)
    """

    def __init__(self) -> None:
        self.extractor: Optional[MediaPipeExtractor] = None
        self.mode_detector: Optional[ModeDetector] = None
        self.letter_predictor: Optional[EnglishLetterPredictor] = None
        self.word_predictor: Optional[EnglishWordPredictor] = None
        self.letter_decoder: Optional[EnglishLetterDecoder] = None
        self.word_decoder: Optional[EnglishWordDecoder] = None
        self.llm_agent: Optional[LLMCorrectionAgent] = None
        self._word_cooldown_ts: float = 0.0

    def load_models(self) -> None:
        """Load all models and initialize components."""
        logger.info("Loading English pipeline models...")

        # MediaPipe
        self.extractor = MediaPipeExtractor(
            model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONF,
            min_tracking_confidence=settings.MEDIAPIPE_MIN_TRACKING_CONF,
            max_num_hands=settings.MEDIAPIPE_MAX_NUM_HANDS,
        )

        # Mode detector
        self.mode_detector = ModeDetector(
            letter_threshold=settings.MOTION_LETTER_THRESHOLD,
            word_threshold=settings.MOTION_WORD_THRESHOLD,
            buffer_frames=settings.MOTION_BUFFER_FRAMES,
        )

        # Letter predictor
        asl_labels = load_label_encoder_from_csv(settings.ASL_LETTER_DATASET_CSV)
        self.letter_predictor = EnglishLetterPredictor(
            model_path=settings.ASL_LETTER_MLP_MODEL,
            labels=asl_labels,
            confidence_threshold=settings.EN_LETTER_CONFIDENCE,
        )

        # Word predictor
        word_classes = load_word_classes(settings.ASL_WORD_CLASSES_CSV)
        vocabulary = load_shared_vocabulary(settings.SHARED_VOCABULARY_CSV)
        self.word_predictor = EnglishWordPredictor(
            model_path=settings.ASL_WORD_LSTM_MODEL,
            word_classes=word_classes,
            vocabulary=vocabulary,
            sequence_length=settings.SEQUENCE_LENGTH,
            confidence_threshold=settings.EN_WORD_CONFIDENCE,
        )

        # Decoders
        self.letter_decoder = EnglishLetterDecoder(
            min_confidence=settings.EN_LETTER_CONFIDENCE,
            stable_window=settings.EN_STABLE_WINDOW,
            majority_ratio=settings.EN_MAJORITY_RATIO,
            cooldown_s=settings.EN_LETTER_COOLDOWN_S,
        )
        self.word_decoder = EnglishWordDecoder()

        # LLM agent
        self.llm_agent = LLMCorrectionAgent(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            confidence_gate=settings.LLM_CONFIDENCE_GATE,
            timeout_ms=settings.LLM_TIMEOUT_MS,
            cache_size=settings.LLM_CACHE_SIZE,
            temperature=settings.LLM_TEMPERATURE,
        )

        logger.info("English pipeline fully loaded.")

    def process_frame(self, rgb_frame: np.ndarray) -> PredictionResult:
        """Process one camera frame through the full pipeline.

        Args:
            rgb_frame: H×W×3 uint8 RGB image.

        Returns:
            ``PredictionResult`` with prediction, sentence, etc.
        """
        # 1. Extract landmarks
        landmarks = self.extractor.extract(rgb_frame)

        # 2. Mode detection
        mode = self.mode_detector.update(landmarks)

        if landmarks is None or mode == SignMode.IDLE:
            return PredictionResult(
                mode=mode.value,
                prediction="",
                confidence=0.0,
                committed="",
                sentence=self._get_full_sentence(),
                event="none",
                llm_corrected=False,
            )

        # 3. Route to appropriate predictor
        if mode == SignMode.LETTER:
            return self._handle_letter(landmarks)
        else:  # WORD
            return self._handle_word(landmarks)

    def _handle_letter(self, landmarks: np.ndarray) -> PredictionResult:
        result = self.letter_predictor.predict(landmarks)
        if result is None:
            return PredictionResult(
                mode="letter", prediction="", confidence=0.0,
                committed="", sentence=self._get_full_sentence(),
                event="none", llm_corrected=False,
            )

        label, conf = result
        decoder_out = self.letter_decoder.update(label, conf)

        # LLM correction on accumulated text
        llm_corrected = False
        if decoder_out["event"] != "none" and self.llm_agent and self.llm_agent.should_correct(conf):
            correction = self.llm_agent.correct_letters(
                LetterCorrectionRequest(raw_text=decoder_out["text"], language="en")
            )
            if correction.was_corrected:
                llm_corrected = True
                # Update decoder text (simplified: just log it)
                logger.info("LLM corrected: '%s' → '%s'", decoder_out["text"], correction.corrected_text)

        return PredictionResult(
            mode="letter",
            prediction=label,
            confidence=conf,
            committed=decoder_out["committed"],
            sentence=self._get_full_sentence(),
            event=decoder_out["event"],
            llm_corrected=llm_corrected,
        )

    def _handle_word(self, landmarks: np.ndarray) -> PredictionResult:
        self.word_predictor.add_frame(landmarks)

        if not self.word_predictor.buffer_full:
            return PredictionResult(
                mode="word", prediction="", confidence=0.0,
                committed="", sentence=self._get_full_sentence(),
                event="none", llm_corrected=False,
            )

        now = time.time()
        if (now - self._word_cooldown_ts) < settings.EN_WORD_COOLDOWN_S:
            return PredictionResult(
                mode="word", prediction="", confidence=0.0,
                committed="", sentence=self._get_full_sentence(),
                event="none", llm_corrected=False,
            )

        candidates = self.word_predictor.predict(top_k=5)
        if not candidates:
            return PredictionResult(
                mode="word", prediction="", confidence=0.0,
                committed="", sentence=self._get_full_sentence(),
                event="none", llm_corrected=False,
            )

        top_word, top_conf = candidates[0]

        # LLM reranking
        llm_corrected = False
        if self.llm_agent and self.llm_agent.should_correct(top_conf):
            correction = self.llm_agent.correct_words(
                WordCorrectionRequest(
                    candidates=candidates,
                    sentence_context=self.word_decoder.get_sentence(),
                    language="en",
                )
            )
            if correction.was_corrected:
                top_word = correction.corrected_text
                llm_corrected = True

        sentence = self.word_decoder.add_word(top_word)
        self._word_cooldown_ts = now
        self.word_predictor.clear_buffer()

        return PredictionResult(
            mode="word",
            prediction=top_word,
            confidence=top_conf,
            committed=top_word,
            sentence=sentence,
            event="append",
            llm_corrected=llm_corrected,
        )

    def _get_full_sentence(self) -> str:
        """Combine letter decoder text + word decoder sentence."""
        parts = []
        if self.letter_decoder and self.letter_decoder.text:
            parts.append(self.letter_decoder.text)
        if self.word_decoder and self.word_decoder.get_sentence():
            parts.append(self.word_decoder.get_sentence())
        return " ".join(parts)

    def reset(self) -> None:
        """Reset all decoder and detector state."""
        if self.letter_decoder:
            self.letter_decoder.reset()
        if self.word_decoder:
            self.word_decoder.clear()
        if self.mode_detector:
            self.mode_detector.reset()
        if self.word_predictor:
            self.word_predictor.clear_buffer()
        self._word_cooldown_ts = 0.0

    def close(self) -> None:
        """Release all resources."""
        if self.extractor:
            self.extractor.close()
