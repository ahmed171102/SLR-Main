# Configuration settings for the SLR API

# Model settings
# For now, we use the existing Arabic MLP. Later, this will point to the GRU model.
MODEL_PATH = r"..\ArSL (Arabic Letters)\arsl_mediapipe_mlp_model_final.h5"

# Stabilization Settings
STABILIZATION_WINDOW_SIZE = 15  # Buffer size for majority vote
STABILIZATION_THRESHOLD = 11    # Minimum matches in buffer (11/15 = ~73% majority)
MIN_CONFIDENCE = 0.75           # Minimum confidence to consider a prediction
HOLD_COOLDOWN_SECONDS = 1.2     # Soft-lock time after committing a letter

# Class Labels (Arabic)
# 28 Letters + 3 Controls
CLASS_LABELS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',
    'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
    'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي',
    'space', 'del', 'nothing'
]
