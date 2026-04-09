"""
coordinator.py — Arabic pipeline inference coordinator.

Mirror of english_pipeline/inference/coordinator.py but:
  - Uses Arabic model paths / thresholds from settings
  - Arabic letter decoder with RTL support
  - Arabic word decoder with RTL support
  - ArSL Word model may not exist yet (graceful fallback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArabicPredictionResult:
    """Container for one frame's Arabic results."""
    language: str = "arabic"
    mode: str = "idle"
    letter_label: Optional[str] = None
    letter_confidence: float = 0.0
    word_label: Optional[str] = None
    word_confidence: float = 0.0
    decoded_text: str = ""
    decoded_text_rtl: str = ""
    current_word: str = ""
    sentence: str = ""
    sentence_rtl: str = ""
    llm_corrected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArabicPipelineCoordinator:
    """Orchestrate the full Arabic recognition pipeline.

    Lifecycle::

        coord = ArabicPipelineCoordinator()
        coord.load_models()
        while ...:
            result = coord.process_frame(rgb_frame)
    """

    def __init__(self) -> None:
        self._loaded = False
        self._letter_predictor = None
        self._word_predictor = None
        self._mobilenet_predictor = None
        self._letter_decoder = None
        self._word_decoder = None
        self._mode_detector = None
        self._extractor = None
        self._llm_agent = None

    def load_models(self) -> None:
        """Load all Arabic models and initialise components."""
        from ..models.letter_predictor import ArabicLetterPredictor
        from ..models.word_predictor import ArabicWordPredictor
        from ..models.mobilenet_predictor import ArabicMobileNetPredictor
        from ..decoders.letter_decoder import ArabicLetterDecoder
        from ..decoders.word_decoder import ArabicWordDecoder
        from ...shared.utils.mediapipe_extractor import MediaPipeExtractor
        from ...shared.utils.mode_detector import ModeDetector
        from ...llm_agent.correction_agent import LLMCorrectionAgent
        from ...config.settings import Settings

        s = Settings

        # Letter predictor
        self._letter_predictor = ArabicLetterPredictor(
            model_path=s.ARSL_LETTER_MLP_PATH,
            label_csv_path=s.ARSL_LETTER_CSV_PATH,
            min_confidence=s.ARABIC_LETTER_CONFIDENCE_THRESHOLD,
        )
        logger.info("Arabic letter MLP loaded")

        # Word predictor (may not exist yet)
        self._word_predictor = ArabicWordPredictor(
            model_path=s.ARSL_WORD_LSTM_PATH,
            class_csv_path=s.ARSL_WORD_CLASS_CSV_PATH,
            vocab_csv_path=s.SHARED_VOCAB_CSV_PATH,
            min_confidence=s.ARABIC_WORD_CONFIDENCE_THRESHOLD,
        )
        if self._word_predictor.is_available:
            logger.info("Arabic word BiLSTM loaded")
        else:
            logger.warning("Arabic word model NOT available — word mode disabled")

        # MobileNet predictor
        self._mobilenet_predictor = ArabicMobileNetPredictor(
            model_path=s.ARSL_LETTER_MOBILENET_PATH,
            label_csv_path=s.ARSL_LETTER_CSV_PATH,
            min_confidence=s.ARABIC_LETTER_CONFIDENCE_THRESHOLD,
        )
        logger.info("Arabic MobileNet loaded")

        # Decoders
        self._letter_decoder = ArabicLetterDecoder(
            min_confidence=s.ARABIC_LETTER_CONFIDENCE_THRESHOLD,
            stable_window=s.STABILITY_WINDOW,
            majority_ratio=s.MAJORITY_RATIO,
            cooldown_s=0.7,  # slightly longer for Arabic
        )
        self._word_decoder = ArabicWordDecoder()

        # Shared components
        self._extractor = MediaPipeExtractor(
            max_hands=s.MP_MAX_HANDS,
            detection_conf=s.MP_DETECTION_CONF,
            tracking_conf=s.MP_TRACKING_CONF,
        )
        self._mode_detector = ModeDetector(
            motion_thresh_letter=s.MOTION_THRESHOLD_LETTER,
            motion_thresh_word=s.MOTION_THRESHOLD_WORD,
            smooth_window=s.MOTION_SMOOTH_WINDOW,
        )
        self._llm_agent = LLMCorrectionAgent(
            api_key=s.LLM_API_KEY,
            model=s.LLM_MODEL,
            confidence_gate=s.LLM_CONFIDENCE_GATE,
        )

        self._loaded = True
        logger.info("Arabic pipeline fully loaded")

    def process_frame(self, rgb_frame: np.ndarray) -> ArabicPredictionResult:
        """Run one RGB frame through the full Arabic pipeline.

        Returns:
            ArabicPredictionResult with letter/word data.
        """
        if not self._loaded:
            raise RuntimeError("Call load_models() before process_frame()")

        result = ArabicPredictionResult()
        landmarks = self._extractor.extract(rgb_frame)

        if landmarks is None:
            result.mode = "idle"
            result.decoded_text = self._letter_decoder.text
            result.decoded_text_rtl = self._letter_decoder.text_rtl
            result.sentence = self._word_decoder.get_sentence()
            result.sentence_rtl = self._word_decoder.get_sentence_rtl()
            return result

        # Detect mode
        mode = self._mode_detector.update(landmarks)
        result.mode = mode.value

        if mode.value == "letter":
            return self._handle_letter(landmarks, result)
        elif mode.value == "word":
            return self._handle_word(landmarks, result)
        else:
            result.decoded_text = self._letter_decoder.text
            result.decoded_text_rtl = self._letter_decoder.text_rtl
            result.sentence = self._word_decoder.get_sentence()
            result.sentence_rtl = self._word_decoder.get_sentence_rtl()
            return result

    def _handle_letter(
        self, landmarks: np.ndarray, result: ArabicPredictionResult
    ) -> ArabicPredictionResult:
        label, conf = self._letter_predictor.predict(landmarks)
        result.letter_label = label
        result.letter_confidence = conf

        decode_out = self._letter_decoder.update(label, conf)
        result.decoded_text = decode_out["text"]
        result.decoded_text_rtl = decode_out["text_rtl"]
        result.current_word = decode_out["word"]

        if decode_out["event"] == "space" and result.current_word:
            corrected = self._llm_agent.correct_letters(
                result.current_word, conf, language="arabic"
            )
            if corrected and corrected.corrected_text != result.current_word:
                result.llm_corrected = True
                result.metadata["llm_original"] = result.current_word
                result.metadata["llm_corrected"] = corrected.corrected_text

        result.sentence = self._word_decoder.get_sentence()
        result.sentence_rtl = self._word_decoder.get_sentence_rtl()
        return result

    def _handle_word(
        self, landmarks: np.ndarray, result: ArabicPredictionResult
    ) -> ArabicPredictionResult:
        if not self._word_predictor or not self._word_predictor.is_available:
            result.metadata["word_model_unavailable"] = True
            result.decoded_text = self._letter_decoder.text
            result.decoded_text_rtl = self._letter_decoder.text_rtl
            result.sentence = self._word_decoder.get_sentence()
            result.sentence_rtl = self._word_decoder.get_sentence_rtl()
            return result

        self._word_predictor.add_frame(landmarks)
        word_result = self._word_predictor.predict()

        if word_result is None:
            result.decoded_text = self._letter_decoder.text
            result.decoded_text_rtl = self._letter_decoder.text_rtl
            result.sentence = self._word_decoder.get_sentence()
            result.sentence_rtl = self._word_decoder.get_sentence_rtl()
            return result

        label, conf, vocab_entry = word_result
        result.word_label = label
        result.word_confidence = conf
        result.metadata["vocab"] = vocab_entry

        corrected = self._llm_agent.correct_words(label, conf, language="arabic")
        if corrected and corrected.corrected_text != label:
            label = corrected.corrected_text
            result.llm_corrected = True

        result.sentence = self._word_decoder.add_word(label)
        result.sentence_rtl = self._word_decoder.get_sentence_rtl()
        result.decoded_text = self._letter_decoder.text
        result.decoded_text_rtl = self._letter_decoder.text_rtl
        return result

    def reset(self) -> None:
        """Reset all decoder state."""
        if self._letter_decoder:
            self._letter_decoder.reset()
        if self._word_decoder:
            self._word_decoder.clear()
        if self._mode_detector:
            self._mode_detector.reset()

    def close(self) -> None:
        """Release resources."""
        if self._extractor:
            self._extractor.close()
        self._loaded = False
