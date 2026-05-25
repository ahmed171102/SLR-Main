import json

NB = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training_Kaggle_Independent.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 1 (Imports) — add proper mediapipe install
new_cell1 = r"""# =========================
# CELL 1: IMPORTS
# =========================

import os
import sys
import time
import warnings
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision

warnings.filterwarnings('ignore', category=UserWarning)

# =============================================
# MediaPipe install — Kaggle does not have it
# pre-installed so we install it quietly here.
# =============================================
print('Installing mediapipe...')
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'mediapipe==0.10.9', '-q', '--no-deps'],
    capture_output=True, text=True
)
if result.returncode != 0:
    # Try without version pin
    result2 = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'mediapipe', '-q'],
        capture_output=True, text=True
    )
    if result2.returncode != 0:
        print('pip install failed:', result2.stderr[:200])
    else:
        print('mediapipe installed (latest version)')
else:
    print('mediapipe 0.10.9 installed')

# Now import
try:
    import mediapipe as mp_lib
    mp_hands_module = mp_lib.solutions.hands
    MEDIAPIPE_AVAILABLE = True
    print(f'MediaPipe : {mp_lib.__version__}')
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f'MediaPipe import failed: {e}')

print('=' * 60)
print('All libraries imported')
print(f'TensorFlow : {tf.__version__}')
print(f'NumPy      : {np.__version__}')
print(f'OpenCV     : {cv2.__version__}')
print(f'MediaPipe  : {"OK - " + mp_lib.__version__ if MEDIAPIPE_AVAILABLE else "FAILED"}')
print('=' * 60)
"""

# Find and replace Cell 1 (code cell, index 1 in cells list — after the markdown title)
code_cell_count = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        code_cell_count += 1
        if code_cell_count == 1:
            nb['cells'][i]['source'] = [new_cell1]
            nb['cells'][i]['outputs'] = []
            print(f"Fixed Cell 1 (imports + mediapipe install) at nb index {i+1}")
            break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook saved!")
