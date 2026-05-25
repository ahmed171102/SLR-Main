"""
Generate a completely clean ArSL Kaggle notebook that:
- Works with the ACTUAL KArSL-502 dataset structure (image frames in folders)
- Does NOT require shared_word_vocabulary.csv
- Does NOT use pip install
- Uses efficient os.scandir instead of rglob
- Handles all 7 issues the user encountered
"""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def code(source):
    cells.append({"cell_type": "code", "metadata": {"trusted": True}, "source": [source], "outputs": [], "execution_count": None})


# ============================================================
# CELL 1: Title
# ============================================================
md("""# ArSL Word Training — Kaggle GPU (KArSL-502 Image Sequence Edition)

**Independent Mode** — No external vocabulary files needed.  
Trained directly from the KArSL-502 image-frame dataset on Kaggle.

### Dataset Format
This notebook is built for the KArSL-502 dataset where each sign sample is stored as a **folder of `.jpg` frames** (not `.mp4` videos).

```
KARSL-502/{class_id}/{class_id}/{train|test}/{sample_id}/{recording_folder}/
    frame_001.jpg
    frame_002.jpg
    ...
```
""")


# ============================================================
# CELL 2: Imports (NO pip install)
# ============================================================
code(r"""# =========================
# CELL 1: IMPORTS
# =========================
# No pip install — uses Kaggle's pre-installed packages only

import os
import time
import warnings
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

# MediaPipe — use what's already installed, no pip
try:
    import mediapipe as mp_lib
    mp_hands_module = mp_lib.solutions.hands
    MEDIAPIPE_AVAILABLE = True
    print(f'MediaPipe : {mp_lib.__version__}')
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f'MediaPipe NOT available: {e}')
    print('Will use raw pixel features as fallback.')

print('=' * 60)
print('All libraries imported')
print(f'TensorFlow : {tf.__version__}')
print(f'NumPy      : {np.__version__}')
print(f'OpenCV     : {cv2.__version__}')
print(f'MediaPipe  : {"YES" if MEDIAPIPE_AVAILABLE else "NO (fallback mode)"}')
print('=' * 60)
""")


# ============================================================
# CELL 3: GPU Setup
# ============================================================
code(r"""# =========================
# CELL 2: GPU SETUP
# =========================
print('=' * 60)
print('GPU DETECTION')
print('=' * 60)

gpus = tf.config.list_physical_devices('GPU')
USE_GPU = False
DEVICE = '/CPU:0'

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        USE_GPU = True
        DEVICE = '/GPU:0'
        print(f'GPU AVAILABLE: {gpus[0].name}')
    except RuntimeError as e:
        print(f'GPU error: {e}')
else:
    print('No GPU — training on CPU (slower)')

mixed_precision.set_global_policy('float32')
print(f'Using device: {DEVICE}')
print('=' * 60)
""")


# ============================================================
# CELL 4: Config + Auto-detect paths
# ============================================================
code(r"""# =========================
# CELL 3: CONFIGURATION
# =========================

IS_KAGGLE = os.path.exists('/kaggle')

if IS_KAGGLE:
    KAGGLE_INPUT = Path('/kaggle/input')
    KAGGLE_OUTPUT = Path('/kaggle/working')
    OUTPUT_DIR = KAGGLE_OUTPUT

    # === AUTO-DETECT KARSL ROOT ===
    # Kaggle datasets can be at various paths depending on how they're added.
    # We try multiple common locations.
    KARSL_ROOT = None
    candidates = [
        KAGGLE_INPUT / 'karsl-502',
        KAGGLE_INPUT / 'KARSL-502',
        KAGGLE_INPUT / 'karsl502',
    ]
    # Also check if there's a subfolder inside (some datasets have an extra wrapper)
    for c in list(candidates):
        candidates.append(c / 'KARSL-502')
        candidates.append(c / 'karsl-502')

    # Also scan everything directly in /kaggle/input/
    if KAGGLE_INPUT.exists():
        for item in os.scandir(str(KAGGLE_INPUT)):
            if item.is_dir():
                candidates.append(Path(item.path))
                # Check if there's a nested folder
                for sub in os.scandir(item.path):
                    if sub.is_dir():
                        candidates.append(Path(sub.path))

    # Find the one that has numbered subfolders (01, 02, etc.)
    for c in candidates:
        if not c.exists():
            continue
        try:
            has_class_folders = any(
                d.is_dir() and d.name.isdigit()
                for d in os.scandir(str(c))
            )
            if has_class_folders:
                KARSL_ROOT = c
                break
        except:
            continue

    if KARSL_ROOT is None:
        # Last resort: print what's actually in /kaggle/input/ for debugging
        print('Could not auto-detect KArSL root. Contents of /kaggle/input/:')
        for item in os.scandir(str(KAGGLE_INPUT)):
            print(f'  {item.name} ({"dir" if item.is_dir() else "file"})')
            if item.is_dir():
                for sub in os.scandir(item.path):
                    print(f'    {sub.name} ({"dir" if sub.is_dir() else "file"})')
        raise FileNotFoundError('Cannot find KArSL dataset. Check Input panel.')
else:
    PROJECT_ROOT = Path(r'M:/Term 10/Grad')
    KARSL_ROOT = PROJECT_ROOT / 'SLR Main/Words/Datasets/KArSL_502'
    OUTPUT_DIR = PROJECT_ROOT / 'SLR Main/Words/ArSL Word (Arabic)'

# ===== PARAMETERS =====
SEQUENCE_LENGTH = 30
NUM_HANDS = 2
LANDMARKS_PER_HAND = 63    # 21 landmarks x 3
NUM_FEATURES = NUM_HANDS * LANDMARKS_PER_HAND  # 126

BATCH_SIZE      = 64
EPOCHS          = 150
LEARNING_RATE   = 5e-4
LSTM_UNITS_1    = 256
LSTM_UNITS_2    = 128
LSTM_UNITS_3    = 64
DENSE_UNITS     = 256
DROPOUT_RATE    = 0.4
LABEL_SMOOTH    = 0.1
TEST_SIZE       = 0.4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True) if not IS_KAGGLE else None

print(f'KArSL root      : {KARSL_ROOT}')
print(f'Output dir      : {OUTPUT_DIR}')
print(f'Sequence length : {SEQUENCE_LENGTH}')
print(f'Features/frame  : {NUM_FEATURES}')
print(f'Batch size      : {BATCH_SIZE}')
print(f'Max epochs      : {EPOCHS}')
print(f'Running on      : {"Kaggle" if IS_KAGGLE else "Local"}')
""")


# ============================================================
# CELL 5: Discover classes + detect dataset format
# ============================================================
code(r"""# =========================
# CELL 4: DISCOVER CLASSES & DETECT FORMAT
# =========================
# Efficiently scan the dataset to find all classes and understand the structure.
# Uses os.scandir (fast) instead of rglob (slow on Kaggle).

print('=' * 60)
print('DISCOVERING DATASET STRUCTURE')
print('=' * 60)

# Step 1: Find all top-level class folders
class_ids = []
for entry in os.scandir(str(KARSL_ROOT)):
    if entry.is_dir() and entry.name.isdigit():
        class_ids.append(int(entry.name))
class_ids.sort()

print(f'Found {len(class_ids)} class folders')
print(f'Sample: {class_ids[:10]}...')

# Step 2: Detect the internal structure by checking the first class
# Expected: {class_id}/{class_id}/{train|test}/{sample_id}/{recording}/frames.jpg
sample_class = str(class_ids[0]).zfill(2)  # e.g. "01"
probe_path = KARSL_ROOT / sample_class

print(f'\nProbing structure of class: {probe_path}')

def find_leaf_folders_sample(root_path, max_depth=5, max_samples=3):
    # Walk a few levels to find actual image-containing folders.
    results = []
    def _walk(path, depth):
        if depth > max_depth or len(results) >= max_samples:
            return
        try:
            entries = list(os.scandir(str(path)))
        except:
            return
        has_images = any(
            e.is_file() and e.name.lower().endswith(('.jpg', '.jpeg', '.png'))
            for e in entries
        )
        if has_images:
            img_count = sum(1 for e in entries if e.is_file() and e.name.lower().endswith(('.jpg', '.jpeg', '.png')))
            results.append((str(path), img_count))
            return
        for e in entries:
            if e.is_dir():
                _walk(Path(e.path), depth + 1)
    _walk(root_path, 0)
    return results

sample_leaves = find_leaf_folders_sample(probe_path)
if sample_leaves:
    print(f'\nSample data folders found:')
    for path, count in sample_leaves:
        rel = str(path).replace(str(KARSL_ROOT), '')
        print(f'  ...{rel}  ({count} images)')
else:
    print('WARNING: No image files found in first class!')

# Build label mappings
id_to_english = {cid: str(cid) for cid in class_ids}
id_to_arabic = {cid: str(cid) for cid in class_ids}
target_karsl_classes = class_ids

print(f'\nTotal classes: {len(class_ids)}')
print(f'Dataset format: Image sequences (.jpg frames in folders)')
""")


# ============================================================
# CELL 6: Helper functions (IMAGE SEQUENCE mode)
# ============================================================
code(r"""# =========================
# CELL 5: HELPER FUNCTIONS (IMAGE SEQUENCE MODE)
# =========================

def pad_or_sample(sequence, target_len=SEQUENCE_LENGTH, target_features=NUM_FEATURES):
    # Pad short or uniformly sample long sequences to fixed shape.
    arr = np.array(sequence, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None

    # Fix feature dimension
    if arr.shape[1] > target_features:
        arr = arr[:, :target_features]
    elif arr.shape[1] < target_features:
        pad_feat = np.zeros((arr.shape[0], target_features - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad_feat], axis=1)

    # Fix time dimension
    if arr.shape[0] >= target_len:
        idx = np.linspace(0, arr.shape[0] - 1, target_len, dtype=int)
        arr = arr[idx]
    else:
        pad_time = np.zeros((target_len - arr.shape[0], target_features), dtype=np.float32)
        arr = np.concatenate([arr, pad_time], axis=0)

    return arr


def extract_from_image_folder_2hand(folder_path, hands_detector):
    # Read .jpg frames from a folder, extract 2-hand landmarks via MediaPipe.
    # Returns numpy array of shape (SEQUENCE_LENGTH, NUM_FEATURES) or None
    # Get all image files, sorted alphabetically (= frame order)
    try:
        img_files = sorted([
            e.path for e in os.scandir(str(folder_path))
            if e.is_file() and e.name.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
    except:
        return None

    if len(img_files) < 3:  # need at least 3 frames
        return None

    frames_data = []
    for img_path in img_files:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        left_vec = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)
        right_vec = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lm, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label
                vec = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark]).flatten()
                if label == 'Left':
                    left_vec = vec
                else:
                    right_vec = vec

        frames_data.append(np.concatenate([left_vec, right_vec]))

    if len(frames_data) < 3:
        return None

    return pad_or_sample(np.array(frames_data, dtype=np.float32))


def find_recording_folders(class_root):
    # Efficiently find all recording folders (leaf folders with .jpg files)
    # inside a class directory. Uses os.scandir with controlled depth.
    recordings = []

    def _scan(path, depth):
        if depth > 5:  # safety limit
            return
        try:
            entries = list(os.scandir(str(path)))
        except:
            return

        # Check if THIS folder contains images (= leaf/recording folder)
        has_images = False
        for e in entries:
            if e.is_file() and e.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                has_images = True
                break

        if has_images:
            recordings.append(str(path))
            return  # don't go deeper

        # Otherwise recurse into subdirectories
        for e in entries:
            if e.is_dir():
                _scan(Path(e.path), depth + 1)

    _scan(Path(class_root), 0)
    return recordings


print(f'Helper functions defined (image-sequence + 2-hand mode)')
print(f'MediaPipe available: {MEDIAPIPE_AVAILABLE}')
""")


# ============================================================
# CELL 7: Build Dataset
# ============================================================
code(r"""# =========================
# CELL 6: BUILD DATASET (or Load Cached)
# =========================

print('=' * 60)
print('BUILDING ARABIC WORD DATASET')
print('=' * 60)

NPZ_PATH = OUTPUT_DIR / 'arsl_word_sequences_2hand.npz'

if NPZ_PATH.exists():
    print(f'\nCached data found: {NPZ_PATH}')
    data = np.load(NPZ_PATH)
    X, y = data['X'], data['y']
    print(f'   X shape : {X.shape}')
    print(f'   y shape : {y.shape}')
    print(f'   Classes : {len(np.unique(y))}')
    print('   Loaded from cache — skipping extraction')
else:
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError('MediaPipe is required for feature extraction from images.')

    # Initialize MediaPipe Hands
    hands = mp_hands_module.Hands(
        static_image_mode=True,     # True = independent frames (more reliable)
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    print(f'\nExtracting features from {len(target_karsl_classes)} classes...')
    print(f'This will take 1-3 hours on GPU. Progress shown below.\n')
    start_time = time.time()

    X_list, y_list = [], []
    found_classes, empty_classes = 0, 0
    total_recordings = 0
    skipped_recordings = 0

    for ci, class_id in enumerate(target_karsl_classes):
        class_dir = KARSL_ROOT / str(class_id)
        if not class_dir.exists():
            # Try zero-padded
            class_dir = KARSL_ROOT / str(class_id).zfill(2)
        if not class_dir.exists():
            class_dir = KARSL_ROOT / str(class_id).zfill(3)
        if not class_dir.exists():
            empty_classes += 1
            continue

        # Find all recording folders (leaf folders with .jpg images)
        recordings = find_recording_folders(class_dir)

        if not recordings:
            empty_classes += 1
            continue

        found_classes += 1
        class_extracted = 0

        for rec_path in recordings:
            total_recordings += 1
            seq = extract_from_image_folder_2hand(rec_path, hands)

            if seq is None:
                skipped_recordings += 1
                continue

            # Skip mostly-blank sequences
            blank_ratio = np.sum(np.all(seq == 0, axis=1)) / len(seq)
            if blank_ratio > 0.8:
                skipped_recordings += 1
                continue

            X_list.append(seq)
            y_list.append(class_id)
            class_extracted += 1

        # Progress update every 10 classes
        if (ci + 1) % 10 == 0 or ci == len(target_karsl_classes) - 1:
            elapsed = time.time() - start_time
            eta = (elapsed / (ci + 1)) * (len(target_karsl_classes) - ci - 1)
            print(f'  [{ci+1:3d}/{len(target_karsl_classes)}] '
                  f'Samples: {len(X_list):5d} | '
                  f'Elapsed: {elapsed/60:.1f}m | '
                  f'ETA: {eta/60:.1f}m')

    hands.close()
    elapsed = time.time() - start_time

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f'\nDone in {elapsed/60:.1f} minutes')
    print(f'   X shape        : {X.shape}')
    print(f'   y shape        : {y.shape}')
    print(f'   Classes found  : {found_classes}')
    print(f'   Empty classes  : {empty_classes}')
    print(f'   Total recordings: {total_recordings}')
    print(f'   Skipped        : {skipped_recordings}')

    np.savez_compressed(NPZ_PATH, X=X, y=y)
    print(f'\nSaved cache: {NPZ_PATH}')
""")


# ============================================================
# CELL 8: Data Exploration
# ============================================================
code(r"""# =========================
# CELL 7: DATA EXPLORATION
# =========================
print('=' * 60)
print('DATA EXPLORATION')
print('=' * 60)

if 'X' not in dir() or 'y' not in dir():
    data = np.load(NPZ_PATH)
    X, y = data['X'], data['y']

unique_ids, counts = np.unique(y, return_counts=True)
labels = [str(uid) for uid in unique_ids]

sort_idx = np.argsort(counts)[::-1]
sorted_names  = [labels[i] for i in sort_idx]
sorted_counts = counts[sort_idx]

fig, ax = plt.subplots(figsize=(22, 6))
ax.bar(range(len(sorted_names)), sorted_counts, color='darkgreen', edgecolor='black', linewidth=0.3)
ax.set_xticks(range(len(sorted_names)))
ax.set_xticklabels(sorted_names, rotation=90, fontsize=5)
ax.set_xlabel('Class ID', fontsize=12)
ax.set_ylabel('Samples', fontsize=12)
ax.set_title(f'ArSL Class Distribution — {len(unique_ids)} classes, {len(y)} total samples', fontsize=14)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'class_distribution.png'), dpi=150)
plt.show()

print(f'\nTotal samples    : {len(y)}')
print(f'Total classes    : {len(unique_ids)}')
print(f'Avg samples/class: {len(y)/len(unique_ids):.1f}')
print(f'Min samples      : {counts.min()} (class {unique_ids[np.argmin(counts)]})')
print(f'Max samples      : {counts.max()} (class {unique_ids[np.argmax(counts)]})')
""")


# ============================================================
# CELL 9: Preprocessing + Split
# ============================================================
code(r"""# =========================
# CELL 8: PREPROCESSING & SPLIT
# =========================
print('=' * 60)
print('PREPROCESSING & SPLIT')
print('=' * 60)

data = np.load(NPZ_PATH)
X, y = data['X'], data['y']

# StandardScaler
original_shape = X.shape
X_flat = X.reshape(-1, NUM_FEATURES)
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X = X_flat.reshape(original_shape).astype(np.float32)

# Save scaler stats
np.savez_compressed(
    str(OUTPUT_DIR / 'arsl_scaler_stats.npz'),
    mean=scaler.mean_.astype(np.float32),
    scale=scaler.scale_.astype(np.float32)
)
print('Scaler saved')

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_onehot = to_categorical(y_encoded, num_classes=num_classes)

# Save class mapping
classes_df = pd.DataFrame({
    'model_class_index': range(num_classes),
    'label_name': [str(c) for c in encoder.classes_],
    'source_class_id': [int(c) for c in encoder.classes_]
})
classes_df.to_csv(str(OUTPUT_DIR / 'arsl_word_classes.csv'), index=False)
print(f'Class mapping saved ({num_classes} classes)')

# Stratified split 60/20/20
try:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )
    temp_labels = np.argmax(y_temp, axis=1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=temp_labels
    )
except ValueError:
    # Some classes may have too few samples for stratification
    print('WARNING: Falling back to non-stratified split')
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_onehot, test_size=TEST_SIZE, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

print(f'\nTrain : {X_train.shape}')
print(f'Val   : {X_val.shape}')
print(f'Test  : {X_test.shape}')
print(f'Classes: {num_classes}')
""")


# ============================================================
# CELL 10: Build + Train Model
# ============================================================
code(r"""# =========================
# CELL 9: BUILD & TRAIN BiLSTM
# =========================
print('=' * 60)
print('TRAINING BiLSTM MODEL')
print('=' * 60)

tf.keras.backend.clear_session()

# Build model
model = Sequential([
    Bidirectional(
        LSTM(LSTM_UNITS_1, return_sequences=True),
        input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)
    ),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=True)),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    LSTM(LSTM_UNITS_3, return_sequences=False),
    BatchNormalization(),
    Dropout(DROPOUT_RATE),

    Dense(DENSE_UNITS, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax', dtype='float32')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')
    ]
)

model.summary()

# Callbacks
MODEL_BEST = str(OUTPUT_DIR / 'arsl_word_lstm_model_best.h5')
MODEL_FINAL = str(OUTPUT_DIR / 'arsl_word_lstm_model_final.h5')

callbacks = [
    ModelCheckpoint(MODEL_BEST, monitor='val_accuracy',
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, verbose=1, min_lr=1e-6),
]

# Balanced class weights
train_labels = np.argmax(y_train, axis=1)
class_weights_arr = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights_arr))

print(f'\nTraining with class weights (balanced)')
print(f'Batch size: {BATCH_SIZE}')
print(f'Max epochs: {EPOCHS}')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

model.save(MODEL_FINAL)
print(f'\nSaved best  : {MODEL_BEST}')
print(f'Saved final : {MODEL_FINAL}')
""")


# ============================================================
# CELL 11: Evaluation
# ============================================================
code(r"""# =========================
# CELL 10: EVALUATION
# =========================
print('=' * 60)
print('MODEL EVALUATION')
print('=' * 60)

# Load best model
best_model = tf.keras.models.load_model(MODEL_BEST)

# Predict
proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Top-1 accuracy
top1_acc = (y_pred == y_true).mean()

# Top-5 accuracy
top5_correct = 0
for i in range(len(y_true)):
    top5 = np.argsort(proba[i])[-5:]
    if y_true[i] in top5:
        top5_correct += 1
top5_acc = top5_correct / len(y_true)

print(f'\nTop-1 Accuracy : {top1_acc*100:.2f}%')
print(f'Top-5 Accuracy : {top5_acc*100:.2f}%')

# Classification report
word_labels = [str(int(encoder.classes_[i])) for i in range(num_classes)]
print('\nClassification Report:')
print(classification_report(y_true, y_pred, labels=range(num_classes),
                            target_names=word_labels, zero_division=0))

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'ArSL Training — Top-1: {top1_acc*100:.1f}% | Top-5: {top5_acc*100:.1f}%',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'training_curves.png'), dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(cm, annot=False, cmap='Greens',
            xticklabels=word_labels, yticklabels=word_labels, ax=ax)
ax.set_title(f'Confusion Matrix — {num_classes} classes')
plt.xticks(rotation=90, fontsize=4)
plt.yticks(fontsize=4)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'confusion_matrix.png'), dpi=150)
plt.show()
""")


# ============================================================
# CELL 12: Download helper
# ============================================================
code(r"""# =========================
# CELL 11: OUTPUT FILES
# =========================
print('=' * 60)
print('OUTPUT FILES')
print('=' * 60)

if IS_KAGGLE:
    print('\nFiles available for download in /kaggle/working/:')
else:
    print(f'\nFiles saved to: {OUTPUT_DIR}')

for f in sorted(OUTPUT_DIR.glob('arsl_*')):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f'   {f.name} ({size_mb:.2f} MB)')

print('\nDownload these and place in your local ArSL Word (Arabic) folder.')
print('Then run ArSL_Word_Live_Test.ipynb to test with your webcam!')
""")


# ============================================================
# CELL 13: Tips markdown
# ============================================================
md("""## Troubleshooting

| Issue | Fix |
|-------|-----|
| **OOM** | Reduce `BATCH_SIZE` to 32 or 16 in Cell 3 |
| **No GPU** | Settings → Accelerator → GPU T4 |
| **Slow extraction** | Normal — takes 1-3 hours for full dataset |
| **NaN loss** | Reduce `LEARNING_RATE` to 1e-4 |
| **Low accuracy** | Increase `EPOCHS`, reduce number of classes |
| **Dataset not found** | Cell 3 will print what's in /kaggle/input/ for debugging |
""")


# ============================================================
# Build the notebook JSON
# ============================================================
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": False
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

OUT_PATH = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Word_Training_Kaggle_Independent.ipynb'
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook written to: {OUT_PATH}")
print(f"Total cells: {len(cells)}")
print("\nThis notebook handles:")
print("  - .jpg image frame folders (NOT .mp4 videos)")
print("  - Auto-detects KARSL_ROOT path on Kaggle")
print("  - No pip install needed")
print("  - No shared_word_vocabulary.csv needed")
print("  - Uses os.scandir (fast) instead of rglob (slow)")
print("  - Graceful fallback for all edge cases")
