"""
Generate ArSL_Keypoints_Training_Kaggle.ipynb — bulletproof version.
Fixes ALL errors encountered in old notebooks.
"""
import json

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":[s]})
def code(s): cells.append({"cell_type":"code","metadata":{"trusted":True},"source":[s],"outputs":[],"execution_count":None})

md("""# ArSL Word Training — KArSL-502 Pre-extracted Keypoints (Fast Mode)
**No MediaPipe. No image processing. No pip install.**  
Loads pre-extracted `.npy` hand keypoints directly → trains BiLSTM.  
Dataset build: **~5 minutes** instead of hours.
""")

# ── CELL 1: Imports ──────────────────────────────────────────────
code(r"""# CELL 1: IMPORTS
import os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision

warnings.filterwarnings('ignore')
print(f'TensorFlow : {tf.__version__}')
print(f'NumPy      : {np.__version__}')
print('Imports OK — No MediaPipe required!')
""")

# ── CELL 2: GPU ──────────────────────────────────────────────────
code(r"""# CELL 2: GPU SETUP
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f'GPU: {gpus[0].name}')
else:
    print('No GPU — running on CPU (training will be slower)')
mixed_precision.set_global_policy('float32')
""")

# ── CELL 3: Config + robust path detection ────────────────────────
code(r"""# CELL 3: CONFIGURATION & PATH DETECTION

IS_KAGGLE   = os.path.exists('/kaggle')
OUTPUT_DIR  = Path('/kaggle/working') if IS_KAGGLE else Path(r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) if not IS_KAGGLE else None

# ── Hyper-parameters ──────────────────────────────────────────────
SEQUENCE_LENGTH = 48     # matches kaleem-app (f_avg=48)
BATCH_SIZE      = 64
EPOCHS          = 150
LEARNING_RATE   = 5e-4
LSTM_UNITS_1    = 256
LSTM_UNITS_2    = 128
LSTM_UNITS_3    = 64
DENSE_UNITS     = 256
DROPOUT_RATE    = 0.4
TEST_SIZE       = 0.4

# ── Auto-detect KARSL root ────────────────────────────────────────
def _has_keypoints(path):
    # Return True if path contains any lh_keypoints subfolder (fast check).
    try:
        for root, dirs, files in os.walk(str(path)):
            if 'lh_keypoints' in dirs:
                return True
            if len(root.split(os.sep)) - len(str(path).split(os.sep)) > 6:
                break  # don't go too deep
    except:
        pass
    return False

KARSL_ROOT = None

if IS_KAGGLE:
    KAGGLE_INPUT = Path('/kaggle/input')
    print('Scanning /kaggle/input/ ...')
    for ds_entry in os.scandir(str(KAGGLE_INPUT)):
        if not ds_entry.is_dir():
            continue
        # Direct check
        if _has_keypoints(ds_entry.path):
            KARSL_ROOT = Path(ds_entry.path)
            break
        # One level deeper (e.g. /kaggle/input/dataset-slug/karsl-502/)
        try:
            for sub in os.scandir(ds_entry.path):
                if sub.is_dir() and _has_keypoints(sub.path):
                    KARSL_ROOT = Path(sub.path)
                    break
        except:
            pass
        if KARSL_ROOT:
            break

    if KARSL_ROOT is None:
        print('\nERROR: Could not find lh_keypoints anywhere in /kaggle/input/')
        print('Contents of /kaggle/input/:')
        for d in os.scandir(str(KAGGLE_INPUT)):
            print(f'  {d.name}/')
        raise FileNotFoundError(
            'lh_keypoints folder not found. '
            'Make sure you added the KArSL-502 dataset with .npy keypoint files.'
        )

    # Labels file
    LABELS_FILE = None
    for root, dirs, files in os.walk(str(KAGGLE_INPUT)):
        for fname in files:
            if 'label' in fname.lower() and fname.endswith('.txt'):
                LABELS_FILE = os.path.join(root, fname)
                break
        if LABELS_FILE:
            break
else:
    KARSL_ROOT  = Path(r'M:\Term 10\Grad\SLR Main\Words\Datasets\KArSL_502')
    LABELS_FILE = str(Path(r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\KARSL-502_Labels.txt'))

print(f'KArSL root   : {KARSL_ROOT}')
print(f'Labels file  : {LABELS_FILE}')
print(f'Output dir   : {OUTPUT_DIR}')
print(f'Sequence len : {SEQUENCE_LENGTH}')
print(f'Running on   : {"Kaggle" if IS_KAGGLE else "Local"}')
""")

# ── CELL 4: Labels ───────────────────────────────────────────────
code(r"""# CELL 4: LOAD LABEL NAMES
id_to_english = {}
id_to_arabic  = {}

if LABELS_FILE and os.path.exists(str(LABELS_FILE)):
    with open(LABELS_FILE, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('SignID'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    sid = int(parts[0])
                    ar  = parts[1].strip()
                    en  = parts[2].strip()
                    id_to_english[sid] = en if en and en not in ('?','??','') else str(sid)
                    id_to_arabic[sid]  = ar if ar and ar not in ('?','??','') else en
                except:
                    continue
    print(f'Labels loaded: {len(id_to_english)} entries')
    print(f'Sample: {list(id_to_english.items())[:5]}')
else:
    print('Labels file not found — numeric IDs will be used as class names')
""")

# ── CELL 5: Discover structure + build recording map ─────────────
code(r"""# CELL 5: DISCOVER DATASET STRUCTURE & BUILD RECORDING MAP
# Handles multiple possible layouts:
#   Layout A: signer/split/class_id/lh_keypoints/     (flat, kaleem-app style)
#   Layout B: group/group/split/class_id/lh_keypoints/ (nested, raw-image style)

print('=' * 60)
print('SCANNING DATASET STRUCTURE')
print('=' * 60)

# Print top-level contents so user can see what's there
top_entries = sorted([e.name for e in os.scandir(str(KARSL_ROOT)) if e.is_dir()])
print(f'Top-level folders ({len(top_entries)}): {top_entries[:10]}...')

class_recordings = {}  # class_id (int) -> list of (lh_npy, rh_npy) pairs
FEATURE_DIM      = None  # detected from first .npy file

def register_class_folder(cls_path):
    # Try to load lh/rh keypoints from a class folder and register them.
    global FEATURE_DIM
    cls_path = Path(cls_path)
    if not cls_path.name.isdigit():
        return
    class_id = int(cls_path.name)

    lh_dir = cls_path / 'lh_keypoints'
    rh_dir = cls_path / 'rh_keypoints'

    if not lh_dir.exists() or not rh_dir.exists():
        return

    lh_files = {Path(p.path).stem: p.path
                for p in os.scandir(str(lh_dir)) if p.name.endswith('.npy')}
    rh_files = {Path(p.path).stem: p.path
                for p in os.scandir(str(rh_dir)) if p.name.endswith('.npy')}

    common = set(lh_files) & set(rh_files)
    if not common:
        return

    if class_id not in class_recordings:
        class_recordings[class_id] = []

    for stem in sorted(common):
        class_recordings[class_id].append((lh_files[stem], rh_files[stem]))
        if FEATURE_DIM is None:
            try:
                arr = np.load(lh_files[stem])
                FEATURE_DIM = arr.shape[-1] if arr.ndim >= 2 else len(arr)
            except:
                pass

# Walk up to 4 levels deep looking for class folders
def scan_for_classes(root, depth=0):
    if depth > 4:
        return
    try:
        entries = list(os.scandir(str(root)))
    except:
        return
    for e in entries:
        if not e.is_dir():
            continue
        if e.name.isdigit() and (Path(e.path) / 'lh_keypoints').exists():
            register_class_folder(e.path)
        else:
            scan_for_classes(e.path, depth + 1)

scan_for_classes(KARSL_ROOT)

if not class_recordings:
    print('\nERROR: No lh_keypoints / rh_keypoints folders found!')
    print('This dataset may only contain raw .jpg images, not pre-extracted .npy keypoints.')
    print('Please use ArSL_Word_Training_Kaggle_Independent.ipynb instead.')
    raise FileNotFoundError('No .npy keypoint files found in dataset.')

# Sort and fill labels
class_ids = sorted(class_recordings.keys())
for cid in class_ids:
    id_to_english.setdefault(cid, str(cid))
    id_to_arabic.setdefault(cid, str(cid))

NUM_FEATURES = (FEATURE_DIM or 21) * 2  # left + right hand

total_recs = sum(len(v) for v in class_recordings.values())
named = sum(1 for c in class_ids if id_to_english[c] != str(c))

print(f'\nClasses found  : {len(class_ids)} (with {named} named labels)')
print(f'Total rec pairs: {total_recs}')
print(f'Avg per class  : {total_recs / max(len(class_ids),1):.1f}')
print(f'Feature dim    : {FEATURE_DIM} per hand → {NUM_FEATURES} total')
print(f'\nFirst 10 classes:')
for cid in class_ids[:10]:
    print(f'  {cid:4d}  {id_to_english[cid]:25s}  {len(class_recordings[cid])} recordings')
""")

# ── CELL 6: Sanity check ─────────────────────────────────────────
code(r"""# CELL 6: SANITY CHECK — verify one full sample loads correctly
print('=' * 60)
print('SANITY CHECK')
print('=' * 60)

_cid = class_ids[0]
_lh, _rh = class_recordings[_cid][0]
print(f'Class {_cid} ({id_to_english[_cid]})')
print(f'LH file : {os.path.basename(_lh)}')
print(f'RH file : {os.path.basename(_rh)}')

_lh_arr = np.load(_lh)
_rh_arr = np.load(_rh)
print(f'LH shape: {_lh_arr.shape}')
print(f'RH shape: {_rh_arr.shape}')

_combined = np.concatenate([_lh_arr, _rh_arr], axis=1) if _lh_arr.ndim > 1 else np.concatenate([_lh_arr, _rh_arr])
print(f'Combined: {_combined.shape}')
print(f'Sample values: {_combined.flatten()[:6]}')
print('SANITY CHECK PASSED!')
""")

# ── CELL 7: Helpers ──────────────────────────────────────────────
code(r"""# CELL 7: HELPER FUNCTIONS

def pad_or_sample(arr, target_len=SEQUENCE_LENGTH, target_feat=NUM_FEATURES):
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Fix features
    if arr.shape[1] > target_feat:
        arr = arr[:, :target_feat]
    elif arr.shape[1] < target_feat:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], target_feat - arr.shape[1]), dtype=np.float32)], axis=1)
    # Fix time
    if arr.shape[0] >= target_len:
        arr = arr[np.linspace(0, arr.shape[0]-1, target_len, dtype=int)]
    else:
        arr = np.concatenate([arr, np.zeros((target_len - arr.shape[0], target_feat), dtype=np.float32)], axis=0)
    return arr  # (SEQUENCE_LENGTH, NUM_FEATURES)


def load_sequence(lh_path, rh_path):
    try:
        lh = np.load(lh_path)
        rh = np.load(rh_path)
    except Exception as e:
        return None
    if lh.ndim == 1: lh = lh.reshape(1, -1)
    if rh.ndim == 1: rh = rh.reshape(1, -1)
    n = min(lh.shape[0], rh.shape[0])
    if n < 3:
        return None
    combined = np.concatenate([lh[:n], rh[:n]], axis=1)
    seq = pad_or_sample(combined)
    blank = np.sum(np.all(seq == 0, axis=1)) / len(seq)
    return None if blank > 0.8 else seq


print(f'Helpers ready | SEQUENCE_LENGTH={SEQUENCE_LENGTH} | NUM_FEATURES={NUM_FEATURES}')
""")

# ── CELL 8: Build dataset ────────────────────────────────────────
code(r"""# CELL 8: BUILD DATASET FROM .NPY KEYPOINTS (or load cache)
print('=' * 60)
print('BUILDING DATASET')
print('=' * 60)

NPZ_PATH = OUTPUT_DIR / 'arsl_word_sequences_keypoints.npz'

if NPZ_PATH.exists():
    print(f'Cache found: {NPZ_PATH}')
    _d = np.load(NPZ_PATH)
    X, y = _d['X'], _d['y']
    print(f'X shape : {X.shape}')
    print(f'y shape : {y.shape}')
    print(f'Classes : {len(np.unique(y))}')
else:
    n_cls  = len(class_ids)
    n_recs = sum(len(v) for v in class_recordings.values())
    print(f'Classes  : {n_cls}')
    print(f'Rec pairs: {n_recs}\n')
    print(f'{"IDX":>5} {"ID":>5} {"Label":<22} {"Recs":>5} {"OK":>5} {"Skip":>5} {"Total":>7} {"Elapsed":>8} {"ETA":>8}')
    print('-' * 82)

    start  = time.time()
    X_list, y_list = [], []
    total_proc = skipped = 0

    for ci, class_id in enumerate(class_ids):
        pairs = class_recordings[class_id]
        label = id_to_english.get(class_id, str(class_id))
        ok = skip = 0

        for lh_path, rh_path in pairs:
            total_proc += 1
            seq = load_sequence(lh_path, rh_path)
            if seq is None:
                skipped += 1; skip += 1
            else:
                X_list.append(seq); y_list.append(class_id); ok += 1

        elapsed = time.time() - start
        rate    = (ci + 1) / elapsed if elapsed > 0 else 1e-9
        eta     = (n_cls - ci - 1) / rate
        print(f'{ci+1:5d} {class_id:5d} {label[:22]:<22} {len(pairs):5d} {ok:5d} {skip:5d} {len(X_list):7d} {elapsed/60:7.1f}m {eta/60:7.1f}m')

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    elapsed = time.time() - start
    print('-' * 82)
    print(f'\nDone in {elapsed:.1f}s ({elapsed/60:.2f} min)')
    print(f'X shape      : {X.shape}')
    print(f'Skipped      : {skipped} / {total_proc}')

    if len(X_list) == 0:
        raise RuntimeError('No samples extracted! Check dataset structure in Cell 5.')

    np.savez_compressed(NPZ_PATH, X=X, y=y)
    print(f'Saved: {NPZ_PATH}')
""")

# ── CELL 9: Preprocess ───────────────────────────────────────────
code(r"""# CELL 9: PREPROCESSING & SPLIT
print('=' * 60)
print('PREPROCESSING & SPLIT')
print('=' * 60)

_d  = np.load(NPZ_PATH)
X, y = _d['X'], _d['y']

# StandardScaler
orig  = X.shape
X_flat = X.reshape(-1, NUM_FEATURES)
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X = X_flat.reshape(orig).astype(np.float32)
np.savez_compressed(str(OUTPUT_DIR / 'arsl_scaler_stats.npz'),
                    mean=scaler.mean_.astype(np.float32),
                    scale=scaler.scale_.astype(np.float32))
print('Scaler saved')

# Encode labels
encoder   = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_onehot  = to_categorical(y_encoded, num_classes=num_classes)

# Save class map with real names
classes_df = pd.DataFrame({
    'model_class_index': range(num_classes),
    'label_name'   : [id_to_english.get(int(encoder.classes_[i]), str(encoder.classes_[i])) for i in range(num_classes)],
    'arabic_name'  : [id_to_arabic.get(int(encoder.classes_[i]),  str(encoder.classes_[i])) for i in range(num_classes)],
    'source_class_id': [int(c) for c in encoder.classes_]
})
classes_df.to_csv(str(OUTPUT_DIR / 'arsl_word_classes.csv'), index=False)
print(f'Class map saved ({num_classes} classes)')
print(classes_df.head(10).to_string())

# 60/20/20 split
try:
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y_onehot, test_size=TEST_SIZE, random_state=42, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=np.argmax(y_tmp,1))
except ValueError:
    print('WARNING: Stratified split failed — using random split')
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y_onehot, test_size=TEST_SIZE, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

print(f'\nTrain : {X_tr.shape} | Val: {X_val.shape} | Test: {X_test.shape}')
print(f'Classes: {num_classes}')
""")

# ── CELL 10: Train ───────────────────────────────────────────────
code(r"""# CELL 10: BUILD & TRAIN BiLSTM
print('=' * 60)
print('TRAINING BiLSTM')
print('=' * 60)

tf.keras.backend.clear_session()

model = Sequential([
    Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True),
                  input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    BatchNormalization(), Dropout(DROPOUT_RATE),
    Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=True)),
    BatchNormalization(), Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS_3, return_sequences=False),
    BatchNormalization(), Dropout(DROPOUT_RATE),
    Dense(DENSE_UNITS, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax', dtype='float32')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
)
model.summary()

MODEL_BEST  = str(OUTPUT_DIR / 'arsl_word_lstm_best.h5')
MODEL_FINAL = str(OUTPUT_DIR / 'arsl_word_lstm_final.h5')

callbacks = [
    ModelCheckpoint(MODEL_BEST, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6),
]

_train_lbl = np.argmax(y_tr, axis=1)
_cw = dict(enumerate(compute_class_weight('balanced', classes=np.unique(_train_lbl), y=_train_lbl)))

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, class_weight=_cw, verbose=1
)
model.save(MODEL_FINAL)
print(f'Saved best  : {MODEL_BEST}')
print(f'Saved final : {MODEL_FINAL}')
""")

# ── CELL 11: Evaluate ────────────────────────────────────────────
code(r"""# CELL 11: EVALUATION
print('=' * 60)
print('EVALUATION')
print('=' * 60)

best = tf.keras.models.load_model(MODEL_BEST)
proba  = best.predict(X_test, verbose=0)
y_pred = np.argmax(proba, axis=1)
y_true = np.argmax(y_test, axis=1)

top1 = (y_pred == y_true).mean()
top5 = sum(1 for i in range(len(y_true)) if y_true[i] in np.argsort(proba[i])[-5:]) / len(y_true)

print(f'Top-1 Accuracy : {top1*100:.2f}%')
print(f'Top-5 Accuracy : {top5*100:.2f}%')

word_labels = [id_to_english.get(int(encoder.classes_[i]), str(encoder.classes_[i])) for i in range(num_classes)]
print(classification_report(y_true, y_pred, target_names=word_labels, zero_division=0))

# Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.suptitle(f'Top-1: {top1*100:.1f}%  Top-5: {top5*100:.1f}%', fontsize=14)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'training_curves.png'), dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(cm, annot=False, cmap='Blues',
            xticklabels=word_labels, yticklabels=word_labels, ax=ax)
ax.set_title(f'Confusion Matrix — {num_classes} classes')
plt.xticks(rotation=90, fontsize=4); plt.yticks(fontsize=4)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'confusion_matrix.png'), dpi=150)
plt.show()
""")

# ── Write notebook ───────────────────────────────────────────────
NB = r'M:\Term 10\Grad\SLR Main\Words\ArSL Word (Arabic)\ArSL_Keypoints_Training_Kaggle.ipynb'
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Notebook: {NB}')
print(f'Cells   : {len(cells)}')
print()
print('Errors from old notebooks — status in new notebook:')
print('  File not found         -> auto-scans /kaggle/input/ for lh_keypoints')
print('  Wrong class count (3)  -> recursive scan finds all 4-digit class folders')
print('  No MediaPipe           -> not needed at all')
print('  pip install failures   -> no pip install')
print('  np.npy files missing   -> clear error message pointing to right notebook')
print('  class_recordings unset -> defined in Cell 5, used in Cells 6/7/8')
print('  NUM_FEATURES unset     -> auto-detected from first .npy file in Cell 5')
print('  Stratified split fail  -> graceful fallback to random split')
print('  No samples extracted   -> RuntimeError with helpful message')
