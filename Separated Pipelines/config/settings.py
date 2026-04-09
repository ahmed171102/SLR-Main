"""
settings.py — Central configuration for the ESHARA SLR System.

All model paths, MediaPipe settings, inference thresholds,
LLM settings, and FastAPI config live here.
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────
# PATH ROOTS
# ──────────────────────────────────────────────────────
# Project root = SLR Main
_THIS_DIR = Path(__file__).resolve().parent          # config/
PIPELINES_ROOT = _THIS_DIR.parent                    # Separated Pipelines/
PROJECT_ROOT = PIPELINES_ROOT.parent                 # SLR Main/

# ──────────────────────────────────────────────────────
# MODEL FILE PATHS
# ──────────────────────────────────────────────────────

# --- English (ASL) Letter Models ---
ASL_LETTER_MLP_MODEL = PROJECT_ROOT / "Letters" / "ASL Letter (English)" / "asl_mediapipe_mlp_model.h5"
ASL_LETTER_MBV2_MODEL = PROJECT_ROOT / "Letters" / "ASL Letter (English)" / "sign_language_model_MobileNetV2_updated.h5"
ASL_LETTER_DATASET_CSV = PROJECT_ROOT / "Letters" / "ASL Letter (English)" / "asl_mediapipe_keypoints_dataset.csv"

# --- Arabic (ArSL) Letter Models ---
ARSL_LETTER_MLP_MODEL = PROJECT_ROOT / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "arsl_mediapipe_mlp_model_final.h5"
ARSL_LETTER_MBV2_MODEL = PROJECT_ROOT / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "mobilenet_arabic_final.h5"
ARSL_LETTER_DATASET_CSV = PROJECT_ROOT / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "FINAL_CLEAN_DATASET.csv"

# --- English (ASL) Word Models ---
ASL_WORD_LSTM_MODEL = PROJECT_ROOT / "Words" / "ASL Word (English)" / "asl_word_lstm_model_best.h5"
ASL_WORD_CLASSES_CSV = PROJECT_ROOT / "Words" / "ASL Word (English)" / "asl_word_classes.csv"

# --- Shared Vocabulary ---
SHARED_VOCABULARY_CSV = PROJECT_ROOT / "Words" / "Shared" / "shared_word_vocabulary.csv"

# --- Arabic (ArSL) Word Models (when trained) ---
ARSL_WORD_LSTM_MODEL = PROJECT_ROOT / "Words" / "ArSL Word (Arabic)" / "arsl_word_lstm_model_best.h5"

# ──────────────────────────────────────────────────────
# MEDIAPIPE SETTINGS
# ──────────────────────────────────────────────────────
MEDIAPIPE_MODEL_COMPLEXITY = 0          # 0 = lite, 1 = full
MEDIAPIPE_MIN_DETECTION_CONF = 0.7
MEDIAPIPE_MIN_TRACKING_CONF = 0.7
MEDIAPIPE_MAX_NUM_HANDS = 2

# ──────────────────────────────────────────────────────
# INFERENCE SETTINGS
# ──────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30      # Word model: number of frames per sequence
NUM_FEATURES = 63         # 21 landmarks × 3 coords (x, y, z)

# ──────────────────────────────────────────────────────
# ENGLISH PIPELINE THRESHOLDS
# ──────────────────────────────────────────────────────
EN_LETTER_CONFIDENCE = 0.80
EN_WORD_CONFIDENCE = 0.35
EN_STABLE_WINDOW = 5
EN_MAJORITY_RATIO = 0.70
EN_LETTER_COOLDOWN_S = 0.6
EN_WORD_COOLDOWN_S = 2.0

# ──────────────────────────────────────────────────────
# ARABIC PIPELINE THRESHOLDS
# ──────────────────────────────────────────────────────
AR_LETTER_CONFIDENCE = 0.85
AR_WORD_CONFIDENCE = 0.35
AR_STABLE_WINDOW = 5
AR_MAJORITY_RATIO = 0.70
AR_LETTER_COOLDOWN_S = 0.7
AR_WORD_COOLDOWN_S = 2.0

# ──────────────────────────────────────────────────────
# MOTION DETECTION (Mode Detector)
# ──────────────────────────────────────────────────────
MOTION_LETTER_THRESHOLD = 0.015
MOTION_WORD_THRESHOLD = 0.030
MOTION_BUFFER_FRAMES = 5

# ──────────────────────────────────────────────────────
# LLM CORRECTION AGENT
# ──────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_CONFIDENCE_GATE = 0.75
LLM_TIMEOUT_MS = 2000
LLM_CACHE_SIZE = 500
LLM_TEMPERATURE = 0.2
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ──────────────────────────────────────────────────────
# FASTAPI SETTINGS
# ──────────────────────────────────────────────────────
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
