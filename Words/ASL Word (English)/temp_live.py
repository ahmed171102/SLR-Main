# ===============================
# CELL 1: IMPORTS & SETUP
# ===============================

import cv2
import json
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from collections import deque

print(f'TensorFlow: {tf.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'MediaPipe: {mp.__version__}')

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'✅ GPU detected: {gpus[0].name}')
else:
    print('⚠️ No GPU — running on CPU')


# ===============================
# CELL 2: CONFIGURATION
# ===============================

from pathlib import Path

PROJECT_ROOT = Path(r"M:/Term 10/Grad")
SLR_MAIN = PROJECT_ROOT / "SLR Main"
WORDS_ROOT = SLR_MAIN / "Words"
OUTPUT_DIR = WORDS_ROOT / "ASL Word (English)"
SHARED_DIR = WORDS_ROOT / "Shared"

# ✅ Model files (verified to exist)
MODEL_PATH = OUTPUT_DIR / "asl_word_lstm_model_best.h5"
CLASSES_CSV = OUTPUT_DIR / "asl_word_classes.csv"
SHARED_CSV = SHARED_DIR / "shared_word_vocabulary.csv"

# Verify files exist
for filepath in [MODEL_PATH, CLASSES_CSV, SHARED_CSV]:
    if not filepath.exists():
        raise FileNotFoundError(f"Required file not found: {filepath}")

# Sequence parameters
SEQUENCE_LENGTH = 30
NUM_FEATURES = None  # Auto-detected from model

# Live inference settings
CONFIDENCE_THRESHOLD = 0.65
STABILITY_WINDOW = 10
PROCESS_EVERY_N_FRAMES = 3

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

print("📂 File verification:")
print(f"   ✅ Model  : {MODEL_PATH.name}")
print(f"   ✅ Classes: {CLASSES_CSV.name}")
print(f"   ✅ Vocab  : {SHARED_CSV.name}")
print(f"\n🎬 Sequence: {SEQUENCE_LENGTH} frames")
print(f"🎯 Confidence: {CONFIDENCE_THRESHOLD}")
print(f"🔁 Stability: {STABILITY_WINDOW} predictions")

# ===============================
# CELL 3: LOAD MODEL & VOCABULARY
# ===============================

# Define custom layer if model uses it
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(x * a, axis=1)
        return output

# Load model
print("Loading model...")
try:
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"TemporalAttention": TemporalAttention}
    )
    print(f"✅ Model loaded: {model.count_params():,} parameters")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Auto-detect feature count
model_input_shape = model.input_shape
NUM_FEATURES = model_input_shape[-1]
NUM_HANDS = 2 if NUM_FEATURES == 126 else 1
print(f"🖐️ Model expects {NUM_FEATURES} features → {NUM_HANDS} hand(s)")

# Load class mapping
print("Loading vocabulary...")
try:
    class_df = pd.read_csv(str(CLASSES_CSV))
    vocab_df = pd.read_csv(str(SHARED_CSV)).dropna(subset=["wlasl_class"])
    
    id_to_english = dict(zip(vocab_df["word_id"].astype(int), vocab_df["english"]))
    
    # Build model_index -> word mapping
    index_to_word = {}
    for _, row in class_df.iterrows():
        idx = int(row["model_class_index"])
        wid = int(row["word_id"])
        index_to_word[idx] = id_to_english.get(wid, f"word_{wid}")
    
    print(f"✅ {len(index_to_word)} word classes loaded")
    print(f"📋 Sample: {list(index_to_word.values())[:10]}")
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print(f"   Make sure these files exist:")
    print(f"   - {CLASSES_CSV}")
    print(f"   - {SHARED_CSV}")
    raise
except Exception as e:
    print(f"❌ Error loading vocabulary: {e}")
    raise

# ==========================================
# CELL 4: LIVE WEBCAM PREDICTION
# ==========================================

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(frame):
    """Extract hand landmarks (63 or 126 features)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    
    left_vec = np.zeros(63, dtype=np.float32)
    right_vec = np.zeros(63, dtype=np.float32)
    draw_landmarks = []
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            vec = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark]).flatten()
            
            if label == "Left":
                left_vec = vec
            else:
                right_vec = vec
            
            draw_landmarks.append(hand_lm)
    
    if NUM_HANDS == 1:
        features = left_vec if np.any(left_vec) else right_vec
    else:
        features = np.concatenate([left_vec, right_vec])
    
    return features, draw_landmarks

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

sequence = []
sentence = []
predictions = []
frame_counter = 0

print("🎥 Starting webcam... Press Q=quit, R=reset, SPACE=space, BACKSPACE=delete")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    
    # Extract features
    features, landmarks = extract_features(frame)
    sequence.append(features)
    sequence = sequence[-SEQUENCE_LENGTH:]  # keep last 30
    
    # Draw hand skeletons
    for hand_lm in landmarks:
        mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
    
    # Run prediction every 3rd frame
    if len(sequence) == SEQUENCE_LENGTH and frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        # Fast inference
        res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()
        predicted_class = np.argmax(res)
        confidence = res[predicted_class]
        predictions.append(predicted_class)
        
        # Stability check
        if len(predictions) >= STABILITY_WINDOW:
            recent = predictions[-STABILITY_WINDOW:]
            if len(set(recent)) == 1 and confidence > CONFIDENCE_THRESHOLD:
                word = index_to_word.get(predicted_class, f"word_{predicted_class}")
                
                # Add to sentence if new
                if len(sentence) == 0 or sentence[-1] != word:
                    sentence.append(word)
                    predictions = []  # reset
        
        # Limit display length
        if len(sentence) > 5:
            sentence = sentence[-5:]
    
    # Display sentence
    cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 40), (245, 117, 16), -1)
    cv2.putText(frame, " ".join(sentence), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("ASL Word Live Test", frame)
    
    # Keyboard controls
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        sentence = []
        predictions = []
        print("🔄 Reset")
    elif key == ord(" "):
        sentence.append(" ")
    elif key == 8:  # BACKSPACE
        if sentence:
            sentence.pop()

cap.release()
cv2.destroyAllWindows()
hands.close()
print("🛑 Closed")

