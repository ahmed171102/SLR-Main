# CELL 1
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


# CELL 2
# ===============================
# CELL 2: CONFIGURATION
# ===============================

PROJECT_ROOT = Path(r'M:/Term 10/Grad')
SLR_MAIN = PROJECT_ROOT / r'Main/Sign-Language-Recognition-System-main/SLR Main'
WORDS_ROOT = SLR_MAIN / r'Words'
OUTPUT_DIR = WORDS_ROOT / r'ASL Word (English)'
SHARED_CSV = WORDS_ROOT / r'Shared/shared_word_vocabulary.csv'

# Model files
MODEL_PATH = OUTPUT_DIR / r'asl_word_lstm_model_final.h5'
CLASSES_CSV = OUTPUT_DIR / r'asl_word_classes.csv'

# Sequence parameters (must match training)
SEQUENCE_LENGTH = 30    # frames per sequence

# Hand detection mode: auto-detected from model input shape
# - 63 features = 1 hand (21 landmarks x 3)
# - 126 features = 2 hands (2 x 21 landmarks x 3)
# Set to None for auto-detection, or override manually:
NUM_FEATURES = None  # will be set after model loads

# Live inference settings
CONFIDENCE_THRESHOLD = 0.35     # minimum confidence to accept a prediction
PREDICTION_INTERVAL = 0.5       # seconds between predictions
STABILITY_WINDOW = 3            # consecutive same predictions needed to confirm
COOLDOWN_TIME = 2.0             # seconds after confirming a word before next

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

print(f'📂 Model  : {MODEL_PATH}')
print(f'📂 Classes: {CLASSES_CSV}')
print(f'🎬 Sequence: {SEQUENCE_LENGTH} frames')
print(f'🎯 Confidence threshold: {CONFIDENCE_THRESHOLD}')
print(f'🔁 Stability window: {STABILITY_WINDOW} predictions')


# CELL 3
import tensorflow as tf
import pandas as pd
import numpy as np
import os

MODEL_WEIGHTS = r'asl_word_lstm_model_final.h5' 
CLASSES_CSV   = r'asl_word_classes.csv'
SHARED_CSV    = r'shared_word_vocabulary.csv'
SCALER_STATS  = r'asl_scaler_stats.npz'

# Load Metadata
class_df = pd.read_csv(CLASSES_CSV)
num_classes = len(class_df)

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63 

# Load Model
print(f"📂 Loading model from {MODEL_WEIGHTS}...")
try:
    model = tf.keras.models.load_model(MODEL_WEIGHTS)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Load Scaler
print(f"📂 Loading scaler from {SCALER_STATS}...")
z = np.load(SCALER_STATS)
scaler_mean = z["mean"].astype(np.float32)
scaler_scale = z["scale"].astype(np.float32)
print("✅ Scaler loaded successfully!")

# Load Vocabulary Mapping
vocab_df = pd.read_csv(SHARED_CSV).dropna(subset=['wlasl_class'])
id_to_english = dict(zip(vocab_df['word_id'].astype(int), vocab_df['english']))
if 'model_class_index' in class_df.columns:
    index_to_word = {int(row['model_class_index']): id_to_english.get(int(row.get('word_id', row['source_class_id'])), row.get('label_name', "Unknown")) 
                     for _, row in class_df.iterrows()}
else:
    # Fallback
    index_to_word = {i: row.get('label_name', 'Unknown') for i, row in class_df.iterrows()}

print(f"🏷️ Loaded {len(index_to_word)} words.")


# CELL 4
# ===============================
# CELL 4: MEDIAPIPE HAND DETECTOR
# ===============================
# Supports both 1-hand and 2-hand detection based on model requirements

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,       # dynamically set based on model
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def extract_landmarks(frame):
    """Extract hand landmarks from a single frame.

    - 1-hand mode (63 features): returns landmarks for the first detected hand.
    - 2-hand mode (126 features): returns concatenated landmarks for both hands.
      If only one hand is detected, the other hand's landmarks are zero-padded.
      Hands are ordered: Left hand first, Right hand second (consistent ordering).

    Returns: (feature_vector, list_of_hand_landmarks_for_drawing)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_landmarks = []

    if NUM_HANDS == 2:
        # Single-hand mode (63 features)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            vec = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
            draw_landmarks = [lm]
            return vec, draw_landmarks
        return np.zeros(NUM_FEATURES, dtype=np.float32), draw_landmarks

    else:
        # Two-hand mode (126 features)
        left_vec = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)
        right_vec = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                draw_landmarks.append(hand_lm)
                label = handedness.classification[0].label  # 'Left' or 'Right'
                vec = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark], dtype=np.float32).flatten()

                # Note: MediaPipe labels are mirrored (camera mirror effect)
                # 'Left' in MediaPipe = right hand in real life (when image is flipped)
                if label == 'Left':
                    left_vec = vec
                else:
                    right_vec = vec

        # Concatenate: [left_hand(63) | right_hand(63)] = 126 features
        combined = np.concatenate([left_vec, right_vec])
        return combined, draw_landmarks

print(f'✅ MediaPipe hand detector ready ({2} hand(s) mode)')
print(f'   Features per frame: {NUM_FEATURES}')


# CELL 5
# ==========================================
# CELL 5
# ==========================================


import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

# ==========================================
# 1. SETUP: 2-HAND MEDIAPIPE INITIALIZATION
# ==========================================
NUM_HANDS = 2
NUM_FEATURES = 63  # 21 landmarks * 3 coordinates (x,y,z) * 2 hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands explicitly for 2 hands
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 2. FEATURE EXTRACTION LOGIC
# ==========================================
def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    landmarks = np.zeros(63)
    hand_lm_list = []

    if results.multi_hand_landmarks:
        # Take the very first hand detected
        hand_landmarks = results.multi_hand_landmarks[0] 
        hand_lm_list.append(hand_landmarks)
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    return landmarks, hand_lm_list



# ==========================================
# 3. LIVE WEBCAM TESTING LOOP
# ==========================================
def run_live_test():
    """Main live testing loop with sliding window prediction."""

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print('❌ Cannot open camera!')
        return

    hand_mode_str = f'{NUM_HANDS} hand(s), {NUM_FEATURES} features'
    print(f'📹 Camera opened [{hand_mode_str}]. Press Q to quit, R to reset, SPACE to add space, BACKSPACE to delete.')

    # --- State variables ---
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=STABILITY_WINDOW)
    sentence_words = []
    current_word = ''
    current_conf = 0.0
    last_prediction_time = 0.0
    last_confirmed_time = 0.0
    hand_detected = False
    hands_count = 0
    fps_history = deque(maxlen=30)

    # Colors
    GREEN = (0, 200, 0)
    RED = (0, 0, 200)
    BLUE = (200, 100, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 220, 220)
    ORANGE = (0, 140, 255)

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- Extract landmarks ---
        landmarks, hand_lm_list = extract_landmarks(frame)
        hand_detected = len(hand_lm_list) > 0
        hands_count = len(hand_lm_list)
        frame_buffer.append(landmarks)

        # --- Draw hand landmarks (all detected hands) ---
        for hand_lm in hand_lm_list:
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # --- Predict when buffer is full ---
        now = time.time()
        if len(frame_buffer) == SEQUENCE_LENGTH and (now - last_prediction_time) >= PREDICTION_INTERVAL:
            last_prediction_time = now

            # Build sequence
            seq = np.array(list(frame_buffer), dtype=np.float32)
            # Scale features
            seq_flat = seq.reshape(-1, NUM_FEATURES)
            seq_scaled = (seq_flat - scaler_mean) / scaler_scale
            seq = seq_scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

            # Check if sequence has enough non-zero frames
            non_zero = np.sum(np.any(seq[0] != 0, axis=1))
            if non_zero >= SEQUENCE_LENGTH * 0.3:  # at least 30% non-zero frames
                # Use direct calling for ultra-fast real-time inference
                proba = model(seq, training=False).numpy()[0] 
                pred_idx = np.argmax(proba)
                pred_conf = proba[pred_idx]
                pred_word = index_to_word.get(pred_idx, '?')
                
                
                
                

                # Top-3 for display
                top3_idx = np.argsort(proba)[-3:][::-1]
                top3 = [(index_to_word.get(i, '?'), proba[i]) for i in top3_idx]

                if pred_conf >= CONFIDENCE_THRESHOLD:
                    current_word = pred_word
                    current_conf = pred_conf
                    prediction_history.append(pred_word)

                    # Check stability: same word predicted N times in a row
                    if (len(prediction_history) == STABILITY_WINDOW and
                        len(set(prediction_history)) == 1 and
                        (now - last_confirmed_time) >= COOLDOWN_TIME):
                        
                        # Confirm the word!
                        sentence_words.append(current_word)
                        last_confirmed_time = now
                        prediction_history.clear()
                        print(f'✅ Confirmed: "{current_word}" ({current_conf:.1%})')
                else:
                    current_word = ''
                    current_conf = 0.0
            else:
                current_word = ''
                current_conf = 0.0

        # --- Draw UI Overlay ---

        # Top bar: prediction info
        cv2.rectangle(frame, (0, 0), (w, 90), BLACK, -1)
        cv2.rectangle(frame, (0, 0), (w, 90), WHITE, 2)

        if current_word:
            color = GREEN if current_conf >= 0.6 else YELLOW if current_conf >= 0.4 else ORANGE
            cv2.putText(frame, f'Word: {current_word}', (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f'Confidence: {current_conf:.1%}', (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Confidence bar
            bar_x = 450
            bar_w = 200
            bar_h = 20
            cv2.rectangle(frame, (bar_x, 20), (bar_x + bar_w, 20 + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * current_conf)
            cv2.rectangle(frame, (bar_x, 20), (bar_x + fill_w, 20 + bar_h), color, -1)
            cv2.rectangle(frame, (bar_x, 20), (bar_x + bar_w, 20 + bar_h), WHITE, 1)

            # Stability progress
            stable_count = sum(1 for p in prediction_history if p == current_word)
            cv2.putText(frame, f'Stability: {stable_count}/{STABILITY_WINDOW}',
                        (bar_x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        else:
            status = 'Show a sign...' if hand_detected else 'No hand detected'
            cv2.putText(frame, status, (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        # Top-3 predictions (right side)
        if current_word and 'top3' in locals():
            tx = w - 320
            cv2.putText(frame, 'Top 3:', (tx, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            for rank, (tw, tc) in enumerate(top3):
                y_pos = 45 + rank * 20
                cv2.putText(frame, f'{rank+1}. {tw} ({tc:.1%})', (tx, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # Bottom bar: sentence
        sentence_text = ' '.join(sentence_words) if sentence_words else '(sentence will appear here)'
        cv2.rectangle(frame, (0, h - 55), (w, h), BLACK, -1)
        cv2.rectangle(frame, (0, h - 55), (w, h), WHITE, 2)
        cv2.putText(frame, f'Sentence: {sentence_text}', (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        # Buffer indicator (bottom-left)
        buf_fill = len(frame_buffer) / SEQUENCE_LENGTH
        buf_color = GREEN if buf_fill >= 1.0 else YELLOW
        cv2.putText(frame, f'Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}',
                    (15, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, buf_color, 1)

        # Hand status indicator (shows hand count for two-hand mode)
        if NUM_HANDS == 2:
            if hands_count == 2:
                hand_color = GREEN
                hand_text = f'HANDS: 2/2'
            elif hands_count == 1:
                hand_color = YELLOW
                hand_text = f'HANDS: 1/2'
            else:
                hand_color = RED
                hand_text = 'NO HANDS'
        else:
            hand_color = GREEN if hand_detected else RED
            hand_text = 'HAND OK' if hand_detected else 'NO HAND'

        cv2.circle(frame, (w - 80, h - 75), 8, hand_color, -1)
        cv2.putText(frame, hand_text, (w - 170, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        # FPS counter
        fps = 1.0 / max(time.time() - frame_start, 1e-6)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(frame, f'FPS: {avg_fps:.0f}', (w - 110, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # Mode indicator
        mode_text = f'Mode: {NUM_HANDS}H / {NUM_FEATURES}F'
        cv2.putText(frame, mode_text, (w - 200, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Cooldown indicator
        cooldown_remaining = max(0, COOLDOWN_TIME - (now - last_confirmed_time))
        if cooldown_remaining > 0:
            cv2.putText(frame, f'Cooldown: {cooldown_remaining:.1f}s',
                        (w // 2 - 80, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)

        # --- Show frame ---
        cv2.imshow('ASL Word Recognition — Live Test', frame)

        # --- Handle keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            sentence_words.clear()
            prediction_history.clear()
            current_word = ''
            print('🔄 Sentence reset')
        elif key == 32:  # SPACE
            sentence_words.append(' ')
            print('   [space added]')
        elif key == 8:   # BACKSPACE
            if sentence_words:
                removed = sentence_words.pop()
                print(f'⬅️ Removed: "{removed}"')

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    final_sentence = ' '.join(sentence_words)
    print(f'\n📝 Final sentence: {final_sentence}')
    return final_sentence

# --- RUN ---
result = run_live_test()
