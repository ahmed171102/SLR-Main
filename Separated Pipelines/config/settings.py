"""
settings.py — Central configuration for both English and Arabic pipelines.

Paths reference the ORIGINAL project folder (SLR Main/) for model files,
so no duplication is needed. Change BASE_PROJECT_DIR if your repo root differs.
"""

import os
from pathlib import Path

# ─────────────────────────── paths ───────────────────────────
# Root of the original project (SLR Main/)
BASE_PROJECT_DIR = Path(os.getenv(
    "SLR_PROJECT_DIR",
    str(Path(__file__).resolve().parents[2])        # …/SLR Main
))

# Root of *this* separated-pipelines package
PIPELINES_DIR = Path(__file__).resolve().parents[1]  # …/Separated Pipelines

# ──────────── Original model / data locations ────────────────
# English (ASL) letter models & data
ASL_LETTER_MLP_MODEL    = BASE_PROJECT_DIR / "Letters" / "ASL Letter (English)" / "asl_mediapipe_mlp_model.h5"
ASL_LETTER_MOBILENET    = BASE_PROJECT_DIR / "Letters" / "ASL Letter (English)" / "sign_language_model_MobileNetV2.h5"
ASL_LETTER_DATASET_CSV  = BASE_PROJECT_DIR / "Letters" / "ASL Letter (English)" / "asl_mediapipe_keypoints_dataset.csv"

# Arabic (ArSL) letter models & data
ARSL_LETTER_MLP_MODEL   = BASE_PROJECT_DIR / "Letters" / "ArSL Letter (Arabic)" / "arsl_mediapipe_mlp_model_best.h5"
ARSL_LETTER_MOBILENET   = BASE_PROJECT_DIR / "Letters" / "ArSL Letter (Arabic)" / "mobilenet_arabic_best_finetuned.h5"
ARSL_LETTER_DATASET_CSV = BASE_PROJECT_DIR / "Letters" / "ArSL Letter (Arabic)" / "Final Notebooks" / "FINAL_CLEAN_DATASET.csv"

# English (ASL) word model
ASL_WORD_MODEL          = BASE_PROJECT_DIR / "Words" / "ASL Word (English)" / "asl_word_lstm_model_best.h5"
ASL_WORD_CLASSES_CSV    = BASE_PROJECT_DIR / "Words" / "ASL Word (English)" / "asl_word_classes.csv"

# Arabic (ArSL) word model (placeholder — train after obtaining KArSL-502)
ARSL_WORD_MODEL         = BASE_PROJECT_DIR / "Words" / "ArSL Word (Arabic)" / "arsl_word_lstm_model_best.h5"

# Shared vocabulary (bilingual bridge)
SHARED_VOCABULARY_CSV   = BASE_PROJECT_DIR / "Words" / "Shared" / "shared_word_vocabulary.csv"

# ─────────────── MediaPipe settings ──────────────────────────
MEDIAPIPE_MODEL_COMPLEXITY      = 0       # 0=lite fastest, 1=full
MEDIAPIPE_MIN_DETECTION_CONF    = 0.7
MEDIAPIPE_MIN_TRACKING_CONF     = 0.7
MEDIAPIPE_MAX_NUM_HANDS         = 2
NUM_LANDMARKS                   = 21
NUM_FEATURES                    = NUM_LANDMARKS * 3   # 63

# ─────────────── Model inference ─────────────────────────────
SEQUENCE_LENGTH = 30       # frames per word sample (BiLSTM window)

# ─────────── English pipeline thresholds ─────────────────────
EN_LETTER_CONFIDENCE_THRESHOLD  = 0.80
EN_WORD_CONFIDENCE_THRESHOLD    = 0.35
EN_LETTER_STABLE_WINDOW         = 5
EN_LETTER_MAJORITY_RATIO        = 0.70
EN_LETTER_COOLDOWN_S            = 0.6
EN_WORD_COOLDOWN_S              = 2.0

# ─────────── Arabic pipeline thresholds ──────────────────────
AR_LETTER_CONFIDENCE_THRESHOLD  = 0.85   # Arabic model is stronger
AR_WORD_CONFIDENCE_THRESHOLD    = 0.35
AR_LETTER_STABLE_WINDOW         = 5
AR_LETTER_MAJORITY_RATIO        = 0.70
AR_LETTER_COOLDOWN_S            = 0.7    # Arabic letters need more spacing
AR_WORD_COOLDOWN_S              = 2.0

# ─────────── Mode detection (letter vs. word) ────────────────
MOTION_THRESHOLD_LETTER = 0.015   # below → still hand → letter mode
MOTION_THRESHOLD_WORD   = 0.030   # above → moving hand → word mode
MOTION_BUFFER_FRAMES    = 5       # frames to average for smoothing

# ─────────── LLM Agent (confidence-gated) ────────────────────
LLM_ENABLED             = True
LLM_PROVIDER            = os.getenv("LLM_PROVIDER", "openai")          # "openai" | "local"
LLM_API_KEY             = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL               = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_CONFIDENCE_GATE     = 0.75    # only call LLM when max softmax < this
LLM_TIMEOUT_MS          = 2000    # max wait for LLM response
LLM_CACHE_SIZE          = 500     # LRU cache entries
LLM_TEMPERATURE         = 0.2     # low temperature for deterministic corrections

# ─────────── FastAPI server ──────────────────────────────────
BACKEND_PORT  = int(os.getenv("BACKEND_PORT", "8000"))
CORS_ORIGINS  = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
