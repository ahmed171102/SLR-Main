"""
mobilenet_predictor.py — English (ASL) MobileNetV2 letter predictor.

Optional fallback model that uses raw images instead of landmarks.
Input: (1, 224, 224, 3) → Output: (label, confidence)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class EnglishMobileNetPredictor:
    """Predict ASL letters from raw camera images using MobileNetV2.

    Args:
        model_path: Path to the MobileNetV2 .h5 model.
        labels: Sorted list of class label strings.
        image_size: Input image size (default 224).
        confidence_threshold: Minimum confidence.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels: List[str],
        image_size: int = 224,
        confidence_threshold: float = 0.80,
    ) -> None:
        from ...shared.models.model_loader import load_keras_model

        self.model = load_keras_model(model_path)
        self.labels = labels
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        logger.info(
            "EnglishMobileNetPredictor ready — %d classes, img=%dx%d",
            len(labels), image_size, image_size,
        )

    def predict(self, bgr_frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """Predict letter from a BGR camera frame.

        Args:
            bgr_frame: OpenCV BGR image (any size).

        Returns:
            ``(label, confidence)`` or ``None`` if below threshold.
        """
        img = cv2.resize(bgr_frame, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf < self.confidence_threshold:
            return None

        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        return label, conf
