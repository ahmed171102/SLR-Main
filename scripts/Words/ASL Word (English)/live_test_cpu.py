import os
import sys

# Fix console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# ========================================
# FORCE CPU MODE - DISABLE ALL GPU/CUDA
# ========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all TF messages

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Must set before importing tensorflow
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from pathlib import Path

# Additional CPU-only configuration
tf.config.set_visible_devices([], 'GPU')

print("[CPU MODE] Running in CPU-only mode (GPU disabled)")
print(f"TensorFlow: {tf.__version__}")
print(f"Devices available: {len(tf.config.list_physical_devices('CPU'))} CPU(s)")

# Configuration
PROJECT_ROOT = Path(r"M:/Term 10/Grad")
SLR_MAIN = PROJECT_ROOT / "SLR Main"
WORDS_ROOT = SLR_MAIN / "Words"
OUTPUT_DIR = WORDS_ROOT / "ASL Word (English)"
SHARED_DIR = WORDS_ROOT / "Shared"

MODEL_PATH = OUTPUT_DIR / "asl_word_lstm_model_final.h5"
CLASSES_CSV = OUTPUT_DIR / "asl_word_classes.csv"
SCALER_STATS = OUTPUT_DIR / "asl_scaler_stats.npz"
SHARED_CSV = SHARED_DIR / "shared_word_vocabulary.csv"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.65
STABILITY_WINDOW = 10


def resolve_model_path(models_dir: Path) -> Path:
    preferred = [
        "asl_word_lstm_model_final.h5",
        "asl_word_lstm_model_best.h5",
    ]

    for name in preferred:
        candidate = models_dir / name
        if candidate.exists():
            return candidate

    # Handle accidental spaces in file names like "final .h5".
    h5_files = sorted(models_dir.glob("*.h5"))
    normalized_lookup = {p.name.replace(" ", "").lower(): p for p in h5_files}
    for name in preferred:
        normalized_name = name.replace(" ", "").lower()
        if normalized_name in normalized_lookup:
            resolved = normalized_lookup[normalized_name]
            print(f"[WARN] Using auto-detected model file: {resolved.name}")
            return resolved

    available = ", ".join(p.name for p in h5_files) if h5_files else "(none found)"
    raise FileNotFoundError(
        f"No compatible model file found in {models_dir}. Available .h5 files: {available}"
    )


MODEL_PATH = resolve_model_path(OUTPUT_DIR)
print(f"[INFO] Using model file: {MODEL_PATH.name}")


# Load model with CPU compatibility
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)


print("Loading model...")
print("[INFO] Model was saved with Keras 3.10 - applying compatibility fixes...")

try:
    # Keras 3.x -> Keras 2.x compatibility
    import json
    import h5py
    
    # Read model config and patch batch_shape issue
    with h5py.File(str(MODEL_PATH), 'r') as f:
        config_str = f.attrs['model_config']
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        
        config = json.loads(config_str)
        
        # Fix batch_shape -> batch_input_shape for Keras 2.x compatibility
        if 'config' in config and 'layers' in config['config']:
            for layer in config['config']['layers']:
                if 'batch_shape' in layer.get('config', {}):
                    layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
    
    # Reconstruct model from fixed config
    model = tf.keras.models.model_from_json(
        json.dumps(config),
        custom_objects={"TemporalAttention": TemporalAttention}
    )
    
    # Load weights separately
    model.load_weights(str(MODEL_PATH))
    print("[OK] Model loaded with Keras 2.x compatibility patches")
    
except Exception as e:
    print(f"[ERROR] Compatibility loading failed: {e}")
    print("\nAttempting direct load...")
    model = tf.keras.models.load_model(
        str(MODEL_PATH), 
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False
    )
    print("[OK] Model loaded")

print("[INFO] Loading scaler...")
try:
    z = np.load(str(SCALER_STATS))
    scaler_mean = z["mean"].astype(np.float32)
    scaler_scale = z["scale"].astype(np.float32)
    print("[OK] Scaler loaded")
except Exception as e:
    print(f"[ERROR] Scaler loading failed: {e}")
    scaler_mean = 0.0
    scaler_scale = 1.0

# Load vocabulary
class_df = pd.read_csv(str(CLASSES_CSV))
vocab_df = pd.read_csv(str(SHARED_CSV)).dropna(subset=["wlasl_class"])
id_to_english = dict(zip(vocab_df["word_id"].astype(int), vocab_df["english"]))
index_to_word = {
    int(row["model_class_index"]): id_to_english.get(
        int(row["word_id"]), f"word_{row['word_id']}"
    )
    for _, row in class_df.iterrows()
}

NUM_FEATURES = model.input_shape[-1]
NUM_HANDS = 2 if NUM_FEATURES == 126 else 1
print(f"[OK] {len(index_to_word)} words loaded, {NUM_HANDS} hand mode")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    left_vec = np.zeros(63, dtype=np.float32)
    right_vec = np.zeros(63, dtype=np.float32)
    landmarks = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label
            vec = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark]).flatten()
            if label == "Left":
                left_vec = vec
            else:
                right_vec = vec
            landmarks.append(hand_lm)

    features = (
        left_vec
        if NUM_HANDS == 1 and np.any(left_vec)
        else right_vec if NUM_HANDS == 1 else np.concatenate([left_vec, right_vec])
    )
    return features, landmarks


# Live test
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sequence = []
sentence = []
predictions = []
frame_counter = 0

print("[CAMERA] Starting webcam... Q=quit, R=reset, SPACE=space")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    features, landmarks = extract_features(frame)
    sequence.append(features)
    sequence = sequence[-SEQUENCE_LENGTH:]

    for hand_lm in landmarks:
        mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

    # Predict every 3 frames
    if len(sequence) == SEQUENCE_LENGTH and frame_counter % 3 == 0:
        # Fast inference (not predict!)
        seq_arr = np.array(sequence, dtype=np.float32)
        seq_flat = seq_arr.reshape(-1, NUM_FEATURES)
        seq_scaled = (seq_flat - scaler_mean) / scaler_scale
        seq_arr = seq_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        res = model(seq_arr, training=False)[0].numpy()
        pred_idx = np.argmax(res)
        conf = res[pred_idx]
        predictions.append(pred_idx)

        if len(predictions) >= STABILITY_WINDOW:
            recent = predictions[-STABILITY_WINDOW:]
            if len(set(recent)) == 1 and conf > CONFIDENCE_THRESHOLD:
                word = index_to_word.get(pred_idx, f"word_{pred_idx}")
                if len(sentence) == 0 or sentence[-1] != word:
                    sentence.append(word)
                    predictions = []

        if len(sentence) > 5:
            sentence = sentence[-5:]

    cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(
        frame,
        " ".join(sentence),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("ASL Word Test (CPU)", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        sentence = []
        predictions = []

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f'[DONE] Final sentence: {" ".join(sentence)}')
