# CELL
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

from IPython.display import display

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


# CELL
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
        "features_per_frame": 258,
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


# CELL
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


# CELL
# If per-language vocab CSV doesn't exist yet, generate it from the shared vocab file.
vocab_path = Path(C["vocab_csv"])
shared_candidates = [
    WORDS_ROOT / "Shared/shared_word_vocabulary.csv",
    WORDS_ROOT / "ASL Word (English)/shared_word_vocabulary.csv",
    WORK_DIR / "shared_word_vocabulary.csv",
]
shared_csv = next((p for p in shared_candidates if p.exists()), None)

if not vocab_path.exists():
    if shared_csv is None:
        raise FileNotFoundError(
            f"Missing vocab file: {vocab_path}. Also couldn't find shared_word_vocabulary.csv in Shared/ or ASL Word (English)/."
        )

    shared = pd.read_csv(shared_csv)
    if LANGUAGE == "asl":
        label_col = "english"
        class_col = "wlasl_class"
    else:
        label_col = "arabic"
        class_col = "karsl_class"

    needed = {"word_id", label_col, class_col}
    missing_shared = needed - set(shared.columns)
    if missing_shared:
        raise ValueError(f"Shared vocab missing required columns: {missing_shared}")

    vocab = shared[["word_id", label_col, class_col]].copy()
    vocab = vocab.dropna(subset=[label_col, class_col])
    vocab = vocab.rename(
        columns={label_col: "label_name", class_col: "source_class_id"}
    )

    vocab["label_name"] = vocab["label_name"].astype(str).str.strip()
    vocab["word_id"] = vocab["word_id"].astype(int)
    vocab["source_class_id"] = vocab["source_class_id"].astype(int)

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.to_csv(vocab_path, index=False)
    print(f"✅ Generated vocab: {vocab_path} (from {shared_csv})")
else:
    vocab = pd.read_csv(vocab_path)

required_cols = {"label_name", "source_class_id"}
missing = required_cols - set(vocab.columns)
if missing:
    raise ValueError(f"Vocab CSV missing required columns: {missing}")

vocab["label_name"] = vocab["label_name"].astype(str).str.strip()
vocab["source_class_id"] = vocab["source_class_id"].astype(int)
if "word_id" in vocab.columns:
    vocab["word_id"] = vocab["word_id"].astype(int)

allowed_class_ids = set(vocab["source_class_id"].tolist())
classid_to_label = dict(zip(vocab["source_class_id"], vocab["label_name"]))

print(f"✅ Loaded vocab rows: {len(vocab)}")
print(f"✅ Allowed classes  : {len(allowed_class_ids)}")
display(vocab.head())


# CELL
mp_holistic = mp.solutions.holistic

def extract_tier3_keypoints(frame_bgr, holistic_obj):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic_obj.process(frame_rgb)
    
    # 1. Pose: 33 x 4 = 132 features
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
    else:
        pose = np.zeros(132, dtype=np.float32)

    # 2. Left hand: 21 x 3 = 63 features
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        has_lh = True
    else:
        lh = np.zeros(63, dtype=np.float32)
        has_lh = False

    # 3. Right hand: 21 x 3 = 63 features
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        has_rh = True
    else:
        rh = np.zeros(63, dtype=np.float32)
        has_rh = False

    # Combine into 258 features
    vec = np.concatenate([pose, lh, rh])
    
    # We consider a hand "detected" if either left or right hand is found
    has_hand = has_lh or has_rh
    
    return vec, has_hand

def to_fixed_sequence(seq, seq_len=30, feat_dim=258):
    n = len(seq)
    if n == 0:
        return np.zeros((seq_len, feat_dim), dtype=np.float32)
    if n >= seq_len:
        idx = np.linspace(0, n - 1, seq_len).astype(int)
        return np.array([seq[i] for i in idx], dtype=np.float32)
    out = np.zeros((seq_len, feat_dim), dtype=np.float32)
    out[:n] = np.array(seq, dtype=np.float32)
    return out


# CELL
samples = []

if C["dataset_type"] == "wlasl":
    wlasl_json = C["wlasl_json"]
    videos_dir = C["videos_dir"]

    print(f"Reading JSON for ASL: {wlasl_json}")
    with open(wlasl_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        class_id = item.get("gloss_id", item.get("class_id", item.get("id", idx)))
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


# CELL
USE_CACHE_IF_EXISTS = False

if USE_CACHE_IF_EXISTS and CACHE_NPZ.exists():
    z = np.load(CACHE_NPZ, allow_pickle=True)
    X = z["X"]
    y_text = z["y_text"] if "y_text" in z else z["y"]
    print(f"✅ Loaded cache: {CACHE_NPZ}")
    print("X:", X.shape, "| y:", y_text.shape)
else:
    X_list, y_list = [], []

    # Using Holistic instead of Hands
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,  # Skip face to speed up processing
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic_obj:
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
                
                # Use our new Tier 3 function!
                vec, has_hand = extract_tier3_keypoints(frame, holistic_obj)
                
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


# CELL
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
    X_tmp, y_tmp, test_size=0.5, random_state=42
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

# Keep word_id if present (makes output compatible with the existing Live Test notebooks/scripts)
if "word_id" in vocab.columns:
    label_to_word = dict(zip(vocab["label_name"], vocab["word_id"]))
    classes_df["word_id"] = classes_df["label_name"].map(label_to_word)

classes_df.to_csv(CLASSES_CSV, index=False)
print("✅ Saved classes:", CLASSES_CSV)
display(classes_df.head())


# CELL
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


# CELL
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


# CELL
loss, acc, top5 = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test acc : {acc:.4f}")
print(f"Top-5 acc: {top5:.4f}")

y_prob = model.predict(X_test, verbose=0)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, labels=np.arange(len(le.classes_)), target_names=le.classes_, zero_division=0))

print("Confusion matrix shape:", confusion_matrix(y_true, y_pred).shape)


# CELL
print("\n✅ Done.")
print(f"Language: {LANGUAGE}")
print("Unified training summary:")
print("- Separate model/cache/classes per language")
print(
    "- If per-language vocab CSV is missing, it is generated from shared_word_vocabulary.csv"
)
print("- No merge notebook required")
