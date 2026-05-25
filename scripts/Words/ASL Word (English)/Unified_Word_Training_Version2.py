# ============================================================
# Unified Word Training Notebook (ASL + ArSL, Independent Vocab)
# ============================================================
# How to use:
# 1) Set LANGUAGE = "asl" or "arsl"
# 2) Update paths in CELL 2
# 3) Ensure vocab CSV for each language exists
# 4) Run all cells top-to-bottom
# ============================================================

# =========================
# CELL 1: IMPORTS
# =========================
import os
import json
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
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

print("✅ Imports loaded")
print("TensorFlow:", tf.__version__)

# =========================
# CELL 2: GLOBAL CONFIG
# =========================
# ---- choose language ----
LANGUAGE = "asl"  # "asl" or "arsl"

# ---- root path ----
PROJECT_ROOT = Path(r"M:/Term 10/Grad")  # <- CHANGE THIS
WORDS_ROOT = PROJECT_ROOT / "SLR Main/Words"

# ---- dataset roots ----
# ASL (WLASL) is currently stored outside the repo in this workspace.
# This matches the existing ASL Word training notebooks.
WLASL_ROOT = PROJECT_ROOT / "Words dataset"
# ArSL (KArSL) is expected under the repo at: SLR Main/Words/Datasets/KArSL_502
WORDS_DATASETS_ROOT = WORDS_ROOT / "Datasets"

# ---- language-specific configuration ----
CFG = {
    "asl": {
        "name": "ASL (English)",
        "work_dir": WORDS_ROOT / "ASL Word (English)",
        "vocab_csv": WORDS_ROOT / "ASL Word (English)/asl_word_vocabulary.csv",
        "dataset_type": "wlasl",
        "wlasl_json": (
            (WLASL_ROOT / "WLASL_v0.3.json")
            if (WLASL_ROOT / "WLASL_v0.3.json").exists()
            else (WORDS_DATASETS_ROOT / "WLASL_v0.3.json")
        ),
        "videos_dir": (
            (WLASL_ROOT / "Words Datasets/WLASL_videos")
            if (WLASL_ROOT / "Words Datasets/WLASL_videos").exists()
            else (WORDS_DATASETS_ROOT / "WLASL_videos")
        ),
        "sequence_len": 30,
        "features_per_frame": 63,
    },
    "arsl": {
        "name": "ArSL (Arabic)",
        "work_dir": WORDS_ROOT / "ArSL Word (Arabic)",
        "vocab_csv": WORDS_ROOT / "ArSL Word (Arabic)/arsl_word_vocabulary.csv",
        "dataset_type": "folder_classid",  # expects class-id folders of videos
        "videos_dir": WORDS_DATASETS_ROOT / "KArSL_502",
        "sequence_len": 30,
        "features_per_frame": 63,
    },
}

if LANGUAGE not in CFG:
    raise ValueError("LANGUAGE must be 'asl' or 'arsl'")

C = CFG[LANGUAGE]
WORK_DIR = C["work_dir"]
WORK_DIR.mkdir(parents=True, exist_ok=True)

CACHE_NPZ = WORK_DIR / f"{LANGUAGE}_word_sequences.npz"
CLASSES_CSV = WORK_DIR / f"{LANGUAGE}_word_classes.csv"
MODEL_BEST = WORK_DIR / f"{LANGUAGE}_word_lstm_model_best.h5"
MODEL_FINAL = WORK_DIR / f"{LANGUAGE}_word_lstm_model_final.h5"
SCALER_STATS = WORK_DIR / f"{LANGUAGE}_scaler_stats.npz"

SEQUENCE_LENGTH = int(C["sequence_len"])
FEATURES_PER_FRAME = int(C["features_per_frame"])

print("=" * 60)
print("RUN CONFIG")
print("=" * 60)
print("LANGUAGE:", LANGUAGE, "|", C["name"])
for k, v in C.items():
    if isinstance(v, Path):
        print(f"{k:16}: {v} {'✅' if v.exists() else '⚠️ missing'}")
    else:
        print(f"{k:16}: {v}")

# =========================
# CELL 3: GPU SETUP
# =========================
print("=" * 60)
print("GPU SETUP")
print("=" * 60)
gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("✅ Memory growth enabled")
    except RuntimeError as e:
        print("⚠️ GPU setup warning:", e)
else:
    print("⚠️ No GPU detected; CPU mode.")

# =========================
# CELL 4: LOAD LANGUAGE VOCAB (NO SHARED FILE)
# =========================
vocab = pd.read_csv(C["vocab_csv"])
required_cols = {"label_name", "source_class_id"}
missing = required_cols - set(vocab.columns)
if missing:
    raise ValueError(f"Vocab CSV missing required columns: {missing}")

vocab["label_name"] = vocab["label_name"].astype(str).str.strip()
vocab["source_class_id"] = vocab["source_class_id"].astype(int)

allowed_class_ids = set(vocab["source_class_id"].tolist())
classid_to_label = dict(zip(vocab["source_class_id"], vocab["label_name"]))

print(f"✅ Loaded vocab rows: {len(vocab)}")
print(f"✅ Allowed classes  : {len(allowed_class_ids)}")
display(vocab.head())

# =========================
# CELL 5: MEDIAPIPE HELPERS
# =========================
mp_hands = mp.solutions.hands


def extract_hand_keypoints(frame_bgr, hands_obj):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands_obj.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Use first detected hand
        lm = results.multi_hand_landmarks[0].landmark
        vec = []
        for p in lm:
            vec.extend([p.x, p.y, p.z])  # 21*3 = 63
        return np.array(vec, dtype=np.float32), True

    return np.zeros((FEATURES_PER_FRAME,), dtype=np.float32), False


def to_fixed_sequence(seq, seq_len=30, feat_dim=63):
    n = len(seq)
    if n == 0:
        return np.zeros((seq_len, feat_dim), dtype=np.float32)
    if n >= seq_len:
        idx = np.linspace(0, n - 1, seq_len).astype(int)
        return np.array([seq[i] for i in idx], dtype=np.float32)
    out = np.zeros((seq_len, feat_dim), dtype=np.float32)
    out[:n] = np.array(seq, dtype=np.float32)
    return out


# =========================
# CELL 6: BUILD SAMPLE LIST
# =========================
samples = []

if C["dataset_type"] == "wlasl":
    wlasl_json = C["wlasl_json"]
    videos_dir = C["videos_dir"]

    with open(wlasl_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        class_id = item.get("gloss_id", item.get("class_id", item.get("id", None)))
        if class_id is None:
            continue
        try:
            class_id = int(class_id)
        except:
            continue

        if class_id not in allowed_class_ids:
            continue

        for inst in item.get("instances", []):
            vid = inst.get("video_id", inst.get("id", None))
            if vid is None:
                continue
            vp = videos_dir / f"{vid}.mp4"
            samples.append(
                {
                    "video_path": vp,
                    "class_id": class_id,
                    "label_name": classid_to_label[class_id],
                }
            )

elif C["dataset_type"] == "folder_classid":
    # Expected structure example:
    # KArSL_502/
    #   0/*.mp4
    #   1/*.mp4
    #   2/*.mp4
    # where folder name == source_class_id
    videos_dir = C["videos_dir"]
    for class_dir in videos_dir.iterdir():
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name)
        except:
            continue
        if class_id not in allowed_class_ids:
            continue

        for vp in class_dir.rglob("*.mp4"):
            samples.append(
                {
                    "video_path": vp,
                    "class_id": class_id,
                    "label_name": classid_to_label[class_id],
                }
            )
else:
    raise ValueError("Unsupported dataset_type in config.")

print("✅ Indexed samples:", len(samples))

# =========================
# CELL 7: EXTRACT OR LOAD CACHE
# =========================
USE_CACHE_IF_EXISTS = True

if USE_CACHE_IF_EXISTS and CACHE_NPZ.exists():
    z = np.load(CACHE_NPZ, allow_pickle=True)
    X = z["X"]
    y_text = z["y_text"]
    print(f"✅ Loaded cache: {CACHE_NPZ}")
    print("X:", X.shape, "| y:", y_text.shape)
else:
    X_list, y_list = [], []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands_obj:
        for s in tqdm(samples, desc=f"Extracting ({LANGUAGE})"):
            vp = s["video_path"]
            if not vp.exists():
                continue

            cap = cv2.VideoCapture(str(vp))
            seq = []
            total_frames = 0
            detected_frames = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                total_frames += 1
                vec, has_hand = extract_hand_keypoints(frame, hands_obj)
                if has_hand:
                    detected_frames += 1
                seq.append(vec)

            cap.release()

            if total_frames == 0:
                continue

            detection_ratio = detected_frames / total_frames
            if detection_ratio < 0.2:
                continue

            seq_fixed = to_fixed_sequence(seq, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
            X_list.append(seq_fixed)
            y_list.append(s["label_name"])

    X = np.array(X_list, dtype=np.float32)
    y_text = np.array(y_list)
    np.savez_compressed(CACHE_NPZ, X=X, y_text=y_text)
    print(f"✅ Saved cache: {CACHE_NPZ}")
    print("X:", X.shape, "| y:", y_text.shape)

if len(X) == 0:
    raise RuntimeError("No extracted samples. Check paths/vocab/dataset format.")

# =========================
# CELL 8: PREPROCESS + SPLIT
# =========================
N, T, F = X.shape
X2 = X.reshape(-1, F)

scaler = StandardScaler()
X2 = scaler.fit_transform(X2)
X_scaled = X2.reshape(N, T, F).astype(np.float32)

np.savez_compressed(
    SCALER_STATS,
    mean=scaler.mean_.astype(np.float32),
    scale=scaler.scale_.astype(np.float32),
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

# Save per-language class mapping
classes_df = pd.DataFrame(
    {"model_class_index": np.arange(len(le.classes_)), "label_name": le.classes_}
)
label_to_src = dict(zip(vocab["label_name"], vocab["source_class_id"]))
classes_df["source_class_id"] = classes_df["label_name"].map(label_to_src)
classes_df.to_csv(CLASSES_CSV, index=False)
print("✅ Saved classes:", CLASSES_CSV)
display(classes_df.head())

# =========================
# CELL 9: BUILD MODEL
# =========================
num_classes = y_train.shape[1]

model = Sequential(
    [
        Bidirectional(
            LSTM(128, return_sequences=True),
            input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME),
        ),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax", dtype="float32"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
    ],
)
model.summary()

# =========================
# CELL 10: TRAIN
# =========================
callbacks = [
    ModelCheckpoint(
        str(MODEL_BEST),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

model.save(MODEL_FINAL)
print("✅ Saved best :", MODEL_BEST)
print("✅ Saved final:", MODEL_FINAL)

# =========================
# CELL 11: EVALUATION
# =========================
loss, acc, top5 = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test acc : {acc:.4f}")
print(f"Top-5 acc: {top5:.4f}")

y_prob = model.predict(X_test, verbose=0)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
print("Confusion matrix shape:", confusion_matrix(y_true, y_pred).shape)

# =========================
# CELL 12: SUMMARY
# =========================
print("\n✅ Done.")
print(f"Language: {LANGUAGE}")
print("Independent pipeline confirmed:")
print("- No shared_word_vocabulary.csv")
print("- No merge notebook required")
print("- Separate vocab/classes/model/cache per language")
