# ============================================================
# ASL Word Training (Independent, No Shared Vocabulary)
# ============================================================
# This script is notebook-friendly: copy cells into .ipynb
# ------------------------------------------------------------

# =========================
# CELL 1: IMPORTS
# =========================
import os
import json
import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

print("✅ Imports done")
print("TensorFlow:", tf.__version__)

# =========================
# CELL 2: GPU CONFIG
# =========================
print("=" * 60)
print("GPU DETECTION")
print("=" * 60)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("✅ Memory growth enabled")
    except RuntimeError as e:
        print("⚠️ GPU setup warning:", e)
else:
    print("⚠️ No GPU found; training will run on CPU")

# =========================
# CELL 3: CONFIG (ASL ONLY)
# =========================
PROJECT_ROOT = Path(r"M:/Term 10/Grad")  # <-- change
WORDS_ROOT   = PROJECT_ROOT / "SLR Main/Words"

ASL_DIR      = WORDS_ROOT / "ASL Word (English)"
DATASETS_DIR = WORDS_ROOT / "Datasets"

ASL_VOCAB_CSV    = ASL_DIR / "asl_word_vocabulary.csv"
WLASL_JSON       = DATASETS_DIR / "WLASL_v0.3.json"
WLASL_VIDEOS_DIR = DATASETS_DIR / "WLASL_videos"

CACHE_NPZ        = ASL_DIR / "asl_word_sequences.npz"
CLASSES_CSV      = ASL_DIR / "asl_word_classes.csv"
MODEL_BEST       = ASL_DIR / "asl_word_lstm_model_best.h5"
MODEL_FINAL      = ASL_DIR / "asl_word_lstm_model_final.h5"
SCALER_NPY       = ASL_DIR / "asl_scaler_stats.npz"

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63  # one-hand landmarks (21 * xyz)

print("=" * 60)
print("ASL-ONLY PATH CHECK")
print("=" * 60)
for n, p in [
    ("ASL vocab", ASL_VOCAB_CSV),
    ("WLASL json", WLASL_JSON),
    ("WLASL videos", WLASL_VIDEOS_DIR),
]:
    print(f"{n:16}: {p} {'✅' if p.exists() else '⚠️ missing'}")

# =========================
# CELL 4: LOAD ASL VOCAB
# =========================
asl_vocab = pd.read_csv(ASL_VOCAB_CSV)
required = {"label_name", "source_class_id"}
missing = required - set(asl_vocab.columns)
if missing:
    raise ValueError(f"Missing required columns in ASL vocab: {missing}")

asl_vocab["label_name"] = asl_vocab["label_name"].astype(str).str.strip()
asl_vocab["source_class_id"] = asl_vocab["source_class_id"].astype(int)

allowed_class_ids = set(asl_vocab["source_class_id"].tolist())
classid_to_label = dict(zip(asl_vocab["source_class_id"], asl_vocab["label_name"]))

print("ASL vocab rows:", len(asl_vocab))
print("ASL classes:", len(allowed_class_ids))
display(asl_vocab.head())

# =========================
# CELL 5: MEDIAPIPE UTILS
# =========================
mp_hands = mp.solutions.hands

def extract_hand_keypoints_from_frame(frame_bgr, hands_obj):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands_obj.process(frame_rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark  # first hand only
        vec = []
        for p in lm:
            vec.extend([p.x, p.y, p.z])
        return np.array(vec, dtype=np.float32), True
    return np.zeros((FEATURES_PER_FRAME,), dtype=np.float32), False

def uniform_sample_or_pad(sequence, target_len=30, feat_dim=63):
    n = len(sequence)
    if n == 0:
        return np.zeros((target_len, feat_dim), dtype=np.float32)
    if n >= target_len:
        idx = np.linspace(0, n - 1, target_len).astype(int)
        return np.array([sequence[i] for i in idx], dtype=np.float32)
    out = np.zeros((target_len, feat_dim), dtype=np.float32)
    out[:n] = np.array(sequence, dtype=np.float32)
    return out

# =========================
# CELL 6: BUILD SAMPLE INDEX FROM WLASL JSON
# =========================
with open(WLASL_JSON, "r", encoding="utf-8") as f:
    wlasl_data = json.load(f)

# Flexible parser for common WLASL json variants
sample_index = []
for item in wlasl_data:
    class_id = item.get("gloss_id", item.get("class_id", item.get("id", None)))
    if class_id is None:
        continue
    try:
        class_id = int(class_id)
    except Exception:
        continue

    if class_id not in allowed_class_ids:
        continue

    instances = item.get("instances", [])
    for inst in instances:
        vid = inst.get("video_id", inst.get("id", None))
        if vid is None:
            continue
        video_path = WLASL_VIDEOS_DIR / f"{vid}.mp4"
        sample_index.append({
            "video_path": video_path,
            "class_id": class_id,
            "label_name": classid_to_label[class_id]
        })

print("Indexed candidate samples:", len(sample_index))

# =========================
# CELL 7: EXTRACT / LOAD CACHE
# =========================
USE_CACHE_IF_EXISTS = True

if USE_CACHE_IF_EXISTS and CACHE_NPZ.exists():
    z = np.load(CACHE_NPZ, allow_pickle=True)
    X = z["X"]
    y_text = z["y_text"]
    print(f"✅ Loaded cache: {CACHE_NPZ}")
    print("X shape:", X.shape, "y:", y_text.shape)
else:
    X_list, y_list = [], []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands_obj:
        for s in tqdm(sample_index, desc="Extracting sequences"):
            vp = s["video_path"]
            if not vp.exists():
                continue

            cap = cv2.VideoCapture(str(vp))
            seq = []
            total = 0
            detected = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                total += 1
                vec, has_hand = extract_hand_keypoints_from_frame(frame, hands_obj)
                if has_hand:
                    detected += 1
                seq.append(vec)

            cap.release()

            if total == 0:
                continue

            detect_ratio = detected / total
            if detect_ratio < 0.2:
                continue

            seq_30 = uniform_sample_or_pad(seq, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
            X_list.append(seq_30)
            y_list.append(s["label_name"])

    X = np.array(X_list, dtype=np.float32)
    y_text = np.array(y_list)

    np.savez_compressed(CACHE_NPZ, X=X, y_text=y_text)
    print(f"✅ Saved cache: {CACHE_NPZ}")
    print("X shape:", X.shape, "y:", y_text.shape)

# =========================
# CELL 8: PREPROCESS + SPLITS
# =========================
if len(X) == 0:
    raise RuntimeError("No samples extracted. Check paths/dataset/vocab mapping.")

# scale features over flattened time axis
N, T, F = X.shape
X_flat = X.reshape(-1, F)

scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X_scaled = X_flat.reshape(N, T, F).astype(np.float32)

np.savez_compressed(
    SCALER_NPY,
    mean=scaler.mean_.astype(np.float32),
    scale=scaler.scale_.astype(np.float32)
)

le = LabelEncoder()
y_idx = le.fit_transform(y_text)
y_onehot = to_categorical(y_idx)

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_scaled, y_onehot, test_size=0.4, random_state=42, stratify=y_idx
)

y_tmp_idx = np.argmax(y_tmp, axis=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp_idx
)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

# save classes mapping
classes_df = pd.DataFrame({
    "model_class_index": np.arange(len(le.classes_)),
    "label_name": le.classes_
})
label_to_src = dict(zip(asl_vocab["label_name"], asl_vocab["source_class_id"]))
classes_df["source_class_id"] = classes_df["label_name"].map(label_to_src)
classes_df.to_csv(CLASSES_CSV, index=False)
print(f"✅ Saved classes map: {CLASSES_CSV}")

# =========================
# CELL 9: MODEL
# =========================
num_classes = y_train.shape[1]

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.2),

    Dense(num_classes, activation="softmax", dtype="float32")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")]
)

model.summary()

# =========================
# CELL 10: TRAIN
# =========================
callbacks = [
    ModelCheckpoint(str(MODEL_BEST), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save(MODEL_FINAL)
print(f"✅ Saved final model: {MODEL_FINAL}")
print(f"✅ Best model path : {MODEL_BEST}")

# =========================
# CELL 11: EVALUATE
# =========================
test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_acc:.4f}")
print(f"Test Top5 : {test_top5:.4f}")

y_prob = model.predict(X_test, verbose=0)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix shape:", cm.shape)

# =========================
# CELL 12: NOTES
# =========================
print("""
Done. This notebook is fully ASL-independent:
- No shared_word_vocabulary.csv
- No Merge_Vocab dependency
- Own vocab/classes/model/cache artifacts only
""")