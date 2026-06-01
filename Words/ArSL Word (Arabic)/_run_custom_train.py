"""Execute custom-words training pipeline (Cells 1-8) after fixes."""
import os
import time
import warnings
from pathlib import Path

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, TimeDistributed,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

# ── Config (Cell 3) ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r'M:/Term 10/Grad')
OUTPUT_DIR = PROJECT_ROOT / 'SLR Main' / 'Words' / 'ArSL Word (Arabic)'
LABELS_FILE = OUTPUT_DIR / 'KARSL-502_Labels.txt'
FULL_NPZ_PATH = OUTPUT_DIR / 'arsl_word_sequences_v2_full.npz'
PARTIAL_NPZ_PATH = OUTPUT_DIR / 'arsl_word_sequences_v2_partial.npz'
CUSTOM_WORDS_CSV = OUTPUT_DIR / 'KARSL-502_BasicWords.csv'
SUBSET_NPZ_PATH = OUTPUT_DIR / 'arsl_custom_subset.npz'
OUTPUT_PREFIX = 'arsl_custom'

POSE_FEATURES = 33 * 4
HAND_FEATURES = 21 * 3
NUM_FEATURES = POSE_FEATURES + HAND_FEATURES * 2
SEQUENCE_LENGTH = 48

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 3e-4
LSTM_UNITS_1 = 192
LSTM_UNITS_2 = 128
LSTM_UNITS_3 = 96
SPATIAL_ENC_1 = 256
SPATIAL_ENC_2 = 192
DENSE_UNITS = 384
DROPOUT_RATE = 0.40
LABEL_SMOOTH = 0.08
GRAD_CLIP_NORM = 1.0
TEST_SIZE = 0.35

SOURCE_NPZ = FULL_NPZ_PATH if FULL_NPZ_PATH.exists() else PARTIAL_NPZ_PATH

# GPU
gpus = tf.config.list_physical_devices('GPU')
DEVICE = '/CPU:0'
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        DEVICE = '/GPU:0'
        print(f'GPU: {gpus[0].name}')
    except RuntimeError as e:
        print(f'GPU setup: {e}')
print(f'Device: {DEVICE}')

# ── Labels (Cell 4) ──────────────────────────────────────────────────────────
id_to_english, id_to_arabic = {}, {}
if LABELS_FILE.exists():
    with open(LABELS_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('SignID'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    sid = int(parts[0])
                    mapped_id = sid + 1
                    en = parts[2].strip()
                    ar = parts[1].strip()
                    id_to_english[mapped_id] = en if en and en not in ('?', '??', '') else str(mapped_id)
                    id_to_arabic[mapped_id] = ar if ar and ar not in ('?', '??', '') else en
                except Exception:
                    pass

# ── Build / load subset (Cell 5) ─────────────────────────────────────────────
if SUBSET_NPZ_PATH.exists():
    print(f'Loading existing subset: {SUBSET_NPZ_PATH}')
    d = np.load(str(SUBSET_NPZ_PATH))
    X, y = d['X'], d['y']
else:
    words_df = pd.read_csv(CUSTOM_WORDS_CSV)
    target_ids = sorted(words_df['class_id'].astype(int).unique().tolist())
    d = np.load(str(SOURCE_NPZ))
    mask = np.isin(d['y'], target_ids)
    X, y = d['X'][mask], d['y'][mask]
    np.savez_compressed(str(SUBSET_NPZ_PATH), X=X, y=y)

print(f'Data: {X.shape}, classes: {len(np.unique(y))}')

# ── Preprocess (Cell 7) ──────────────────────────────────────────────────────
orig_shape = X.shape
X_flat = X.reshape(-1, NUM_FEATURES)
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X = X_flat.reshape(orig_shape).astype(np.float32)

np.savez_compressed(
    str(OUTPUT_DIR / f'{OUTPUT_PREFIX}_scaler.npz'),
    mean=scaler.mean_.astype(np.float32),
    scale=scaler.scale_.astype(np.float32),
)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
y_onehot = to_categorical(y_encoded, num_classes=num_classes)

class_df = pd.DataFrame({
    'model_class_index': range(num_classes),
    'karsl_class_id': encoder.classes_.tolist(),
    'english': [id_to_english.get(int(c), str(c)) for c in encoder.classes_],
    'arabic': [id_to_arabic.get(int(c), str(c)) for c in encoder.classes_],
})
class_df.to_csv(OUTPUT_DIR / f'{OUTPUT_PREFIX}_classes.csv', index=False)

try:
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_onehot, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=np.argmax(y_tmp, 1)
    )
except ValueError:
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_onehot, test_size=TEST_SIZE, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

train_ints = np.argmax(y_train, axis=1)
cw_array = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_ints)
class_weights = dict(enumerate(np.clip(cw_array, 0.5, 10.0)))
print(f'Train {X_train.shape[0]} | Val {X_val.shape[0]} | Test {X_test.shape[0]} | Classes {num_classes}')

# ── Train (Cell 8) ───────────────────────────────────────────────────────────
tf.keras.backend.clear_session()

POSE_F = tf.constant(POSE_FEATURES, dtype=tf.int32)
HAND_F = tf.constant(HAND_FEATURES, dtype=tf.int32)

def augment_sequence(x, y_label):
    x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.007)
    x = tf.roll(x, shift=tf.random.uniform([], -4, 5, dtype=tf.int32), axis=0)
    mask = tf.cast(tf.random.uniform([SEQUENCE_LENGTH, 1]) > 0.12, tf.float32)
    x = x * mask
    x = x * tf.random.uniform([], 0.9, 1.1)

    def flip_hands():
        pose = x[:, :POSE_F]
        lh = x[:, POSE_F:POSE_F + HAND_F]
        rh = x[:, POSE_F + HAND_F:]
        return tf.concat([pose, rh, lh], axis=-1)

    x = tf.cond(tf.random.uniform([]) > 0.5, flip_hands, lambda: x)
    return x, y_label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(min(len(X_train), 8000), seed=42)
    .map(augment_sequence, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)

inputs = Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES), name='landmark_input')
x = TimeDistributed(Dense(SPATIAL_ENC_1, activation='relu'))(inputs)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Dropout(0.2))(x)
x = TimeDistributed(Dense(SPATIAL_ENC_2, activation='relu'))(x)
x = TimeDistributed(BatchNormalization())(x)
x = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True))(x)
x = BatchNormalization()(x)
x = tf.keras.layers.SpatialDropout1D(DROPOUT_RATE)(x)
x = Bidirectional(LSTM(LSTM_UNITS_2, return_sequences=True))(x)
x = BatchNormalization()(x)
x = tf.keras.layers.SpatialDropout1D(DROPOUT_RATE)(x)
attn_out, _ = tf.keras.layers.MultiHeadAttention(
    num_heads=4, key_dim=LSTM_UNITS_2 // 4, dropout=0.1
)(x, x, return_attention_scores=True)
x = tf.keras.layers.Add()([x, attn_out])
x = tf.keras.layers.LayerNormalization()(x)
x = LSTM(LSTM_UNITS_3, return_sequences=False)(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(DENSE_UNITS, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(DENSE_UNITS // 2, activation='relu')(x)
x = Dropout(DROPOUT_RATE * 0.5)(x)
outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs, outputs, name='ArSL_100Words_GradProject')
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=GRAD_CLIP_NORM)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')],
)
model.summary()

MODEL_BEST = str(OUTPUT_DIR / f'{OUTPUT_PREFIX}_best.h5')
MODEL_FINAL = str(OUTPUT_DIR / f'{OUTPUT_PREFIX}_final.h5')
callbacks = [
    ModelCheckpoint(MODEL_BEST, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.TerminateOnNaN(),
]

print(f'Training on {DEVICE} ...')
t0 = time.time()
with tf.device(DEVICE):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
elapsed = (time.time() - t0) / 60
best = max(history.history['val_accuracy'])
print(f'Done in {elapsed:.1f} min | best val_acc {best:.4f} | epochs run {len(history.history["loss"])}')
model.save(MODEL_FINAL)
print(f'Saved: {MODEL_BEST}')
