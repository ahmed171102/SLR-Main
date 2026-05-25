"""
Quick webcam test for best_cslr_model.weights (1).h5
Run: python webcam_test.py

Requirements:
    pip install tensorflow mediapipe opencv-python h5py numpy

Place in the same folder as:
    - best_cslr_model.weights (1).h5   (required)
    - vocab.json                        (optional — shows [IDX:N] if missing)
"""

import os, json, sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    LSTM, Bidirectional, MultiHeadAttention,
    LayerNormalization, Softmax,
)
from tensorflow.keras.models import Model
from collections import deque

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH    = os.path.join(SCRIPT_DIR, "best_cslr_model.weights (1).h5")
VOCAB_PATH      = os.path.join(SCRIPT_DIR, "vocab.json")
SEQUENCE_LENGTH = 424
NUM_FEATURES    = 232
MAX_TOKENS      = 3000
VOCAB_SIZE_CTC  = MAX_TOKENS + 1   # 3001 (index 0 = CTC blank)
INFER_EVERY     = 5                # run inference every N frames

# ── 1. Custom CTC layer (needed for loading the saved model) ─────────────────
class CTCLossLayer(tf.keras.layers.Layer):
    def __init__(self, blank_index=0, **kwargs):
        super().__init__(**kwargs)
        self.blank_index = blank_index

    def call(self, inputs):
        y_pred, labels, input_length, label_length = inputs
        log_probs   = tf.math.log(y_pred + 1e-8)
        log_probs   = tf.transpose(log_probs, [1, 0, 2])
        input_len   = tf.cast(tf.reshape(input_length, [-1]), tf.int32)
        label_len   = tf.cast(tf.reshape(label_length, [-1]), tf.int32)
        label_len   = tf.minimum(label_len, input_len)
        label_len   = tf.maximum(label_len, 1)
        loss = tf.nn.ctc_loss(
            labels=labels, logits=log_probs,
            label_length=label_len, logit_length=input_len,
            logits_time_major=True, blank_index=self.blank_index,
        )
        return tf.reduce_mean(tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss)))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"blank_index": self.blank_index})
        return cfg


# ── 2. Load vocabulary (optional) ────────────────────────────────────────────
vocab = None
if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)           # list of words, index 0 = blank/"" etc.
    print(f"[OK] Vocab loaded  — {len(vocab)} tokens from {VOCAB_PATH}")
else:
    print(f"[WARN] vocab.json not found at {VOCAB_PATH}")
    print("       Predictions will show raw indices like [IDX:42].")
    print("       Add vocab.json next to this script to see real words.\n")


def decode_indices(indices):
    """Convert CTC-decoded index array to a sentence string."""
    words = []
    for idx in indices:
        if idx < 0 or idx == 0:        # blank or padding
            continue
        if vocab is not None and idx < len(vocab):
            words.append(vocab[idx])
        else:
            words.append(f"[IDX:{idx}]")
    return " ".join(words) if words else ""


# ── 3. Build & load inference model ──────────────────────────────────────────
print(f"[..] Loading model from:\n     {WEIGHTS_PATH}")
if not os.path.exists(WEIGHTS_PATH):
    print(f"[ERR] Weight file not found! Put the .h5 file next to this script.")
    sys.exit(1)

# Rebuild architecture (mirrors how2sign (1).ipynb cell 8)
inputs  = Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name="input")
x       = Bidirectional(LSTM(256, return_sequences=True), name="bilstm_1")(inputs)
x       = BatchNormalization(name="bn_1")(x)
attn    = MultiHeadAttention(num_heads=4, key_dim=64, name="mha")(x, x)
x       = LayerNormalization(name="ln_1")(x + attn)
x       = Dropout(0.3, name="drop_1")(x)
x       = Bidirectional(LSTM(256, return_sequences=True), name="bilstm_2")(x)
x       = BatchNormalization(name="bn_2")(x)
x       = Dropout(0.3, name="drop_2")(x)
logits  = Dense(VOCAB_SIZE_CTC, name="logits")(x)
y_pred  = Softmax(name="prediction")(logits)

model_infer = Model(inputs=inputs, outputs=y_pred, name="ctc_inference")

# Load weights — works whether the .h5 is weights-only or full-model
try:
    model_infer.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    print("[OK] Weights loaded (by_name).")
except Exception as e:
    print(f"[WARN] by_name load failed ({e}), trying full model load...")
    try:
        full = tf.keras.models.load_model(
            WEIGHTS_PATH,
            custom_objects={"CTCLossLayer": CTCLossLayer},
            compile=False,
        )
        # copy shared layer weights
        for layer in model_infer.layers:
            try:
                src = full.get_layer(layer.name)
                layer.set_weights(src.get_weights())
            except Exception:
                pass
        print("[OK] Weights loaded (full model copy).")
    except Exception as e2:
        print(f"[ERR] Could not load weights: {e2}")
        sys.exit(1)


# ── 4. MediaPipe feature extraction (232-dim) ─────────────────────────────────
import mediapipe as mp

_MP_HOLISTIC   = mp.solutions.holistic
_MP_DRAWING    = mp.solutions.drawing_utils

_FACE_KP_INDICES = (
    [17,18,19,20,21] + [22,23,24,25,26] +
    [36,37,38,39,40,41] + [42,43,44,45,46,47] +
    [68,69] +
    [27,28,29,30] + [33] +
    [48,49,50,51,52,53,54,55,56,57,58,59] +
    [60,61,62,63,64,65,66,67]
)

_MP_FACE_TO_OP70 = {
    17:70, 18:63, 19:105, 20:66, 21:107,
    22:336, 23:296, 24:334, 25:293, 26:300,
    36:33, 37:160, 38:158, 39:133, 40:153, 41:144,
    42:362, 43:385, 44:387, 45:263, 46:373, 47:380,
    68:468, 69:473,
    27:6, 28:197, 29:195, 30:5, 33:1,
    48:61, 49:185, 50:40, 51:39, 52:37, 53:0,
    54:267, 55:269, 56:270, 57:409, 58:291, 59:375,
    60:78, 61:191, 62:80, 63:81,
    64:311, 65:310, 66:415, 67:308,
}


def mediapipe_to_232dim(results):
    """MediaPipe Holistic → (232,) float32: body(50)+lhand(42)+rhand(42)+face(98)"""
    pose = np.zeros((25, 2), dtype=np.float32)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        def sp(oi, mi):
            if mi < len(lm): pose[oi] = [lm[mi].x, lm[mi].y]
        sp(0,0); sp(2,12); sp(3,14); sp(4,16); sp(5,11); sp(6,13); sp(7,15)
        sp(9,24); sp(10,26); sp(11,28); sp(12,23); sp(13,25); sp(14,27)
        sp(15,5); sp(16,2); sp(17,8); sp(18,7); sp(19,31); sp(21,29)
        sp(22,32); sp(24,30)
        if pose[2].any() and pose[5].any():
            pose[1] = (pose[2] + pose[5]) / 2.0
        if pose[9].any() and pose[12].any():
            pose[8] = (pose[9] + pose[12]) / 2.0

    lh = np.zeros((21, 2), dtype=np.float32)
    if results.left_hand_landmarks:
        for k, pt in enumerate(results.left_hand_landmarks.landmark):
            lh[k] = [pt.x, pt.y]

    rh = np.zeros((21, 2), dtype=np.float32)
    if results.right_hand_landmarks:
        for k, pt in enumerate(results.right_hand_landmarks.landmark):
            rh[k] = [pt.x, pt.y]

    face = np.zeros((49, 2), dtype=np.float32)
    if results.face_landmarks:
        mesh = results.face_landmarks.landmark
        for k, op_idx in enumerate(_FACE_KP_INDICES):
            mi = _MP_FACE_TO_OP70.get(op_idx)
            if mi is not None and mi < len(mesh):
                face[k] = [mesh[mi].x, mesh[mi].y]

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten(), face.flatten()])


# ── 5. Stabilisation tracker ──────────────────────────────────────────────────
class StabilizationTracker:
    def __init__(self, window=15, ratio=0.6):
        self.window = deque(maxlen=window)
        self.ratio  = ratio
        self.stable = ""

    def update(self, text):
        self.window.append(text)
        if not self.window: return
        from collections import Counter
        counts = Counter(self.window)
        top, n = counts.most_common(1)[0]
        if n / len(self.window) >= self.ratio:
            self.stable = top


# ── 6. Webcam loop ────────────────────────────────────────────────────────────
def run_webcam(camera_id=0):
    print(f"\n[..] Opening camera {camera_id} …  Press Q to quit, R to reset buffer.")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERR] Cannot open camera {camera_id}")
        return

    tracker      = StabilizationTracker(window=15, ratio=0.6)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    frame_count  = 0
    last_text    = ""

    with _MP_HOLISTIC.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed — camera disconnected?")
                break

            # MediaPipe inference
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Draw hand/pose landmarks
            _MP_DRAWING.draw_landmarks(bgr, results.left_hand_landmarks,
                                       mp.solutions.hands.HAND_CONNECTIONS)
            _MP_DRAWING.draw_landmarks(bgr, results.right_hand_landmarks,
                                       mp.solutions.hands.HAND_CONNECTIONS)

            # Collect features
            feat = mediapipe_to_232dim(results)
            frame_buffer.append(feat)
            frame_count += 1

            # Run model every INFER_EVERY frames and when buffer has enough data
            if len(frame_buffer) >= 15 and frame_count % INFER_EVERY == 0:
                seq = np.stack(frame_buffer)                  # (T, 232)
                T   = seq.shape[0]
                if T < SEQUENCE_LENGTH:
                    pad = np.zeros((SEQUENCE_LENGTH - T, NUM_FEATURES), np.float32)
                    seq = np.vstack([seq, pad])

                X         = seq[np.newaxis]                   # (1, 424, 232)
                preds     = model_infer(X, training=False)    # (1, 424, 3001)
                inp_lens  = np.array([min(T, SEQUENCE_LENGTH)])
                decoded, _= tf.keras.backend.ctc_decode(preds, inp_lens, greedy=True)
                indices   = decoded[0][0].numpy()
                sentence  = decode_indices(indices)
                tracker.update(sentence)
                last_text = tracker.stable

            # ── HUD ──────────────────────────────────────────
            h, w = bgr.shape[:2]
            buf_pct = int(len(frame_buffer) / SEQUENCE_LENGTH * 100)

            # dark band at top
            cv2.rectangle(bgr, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.putText(bgr, last_text or "(waiting for prediction...)",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # buffer fill bar
            cv2.rectangle(bgr, (0, h-10), (int(w * buf_pct / 100), h), (0, 200, 0), -1)
            cv2.putText(bgr, f"Buffer {buf_pct}%  |  Q=quit  R=reset",
                        (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Sign Language — Quick Test", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                frame_buffer.clear()
                tracker.stable = ""
                last_text = ""
                print("[INFO] Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Done.")


if __name__ == "__main__":
    camera = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam(camera)
