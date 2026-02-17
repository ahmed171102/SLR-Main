# Words Pipeline — Full Architecture & Data Flow

> **Team Reference** — How data moves from raw video → trained model → prediction

---

## End-to-End Pipeline

```
                          ┌──────────────────────────┐
                          │     RAW VIDEO FILES       │
                          │  (.mp4, 2-10 sec each)    │
                          │  WLASL: 11,980 videos     │
                          │  KArSL: TBD               │
                          └────────────┬───────────────┘
                                       │
                          ┌────────────▼───────────────┐
                          │  SHARED VOCABULARY FILTER   │
                          │  shared_word_vocabulary.csv  │
                          │  157 matched bilingual words │
                          │  Only train on these words  │
                          └────────────┬───────────────┘
                                       │
                     ┌─────────────────┼──────────────────┐
                     │                                     │
          ┌──────────▼───────────┐           ┌─────────────▼──────────┐
          │   ASL (English)      │           │   ArSL (Arabic)         │
          │   WLASL videos       │           │   KArSL videos/keypts   │
          └──────────┬───────────┘           └─────────────┬──────────┘
                     │                                     │
          ┌──────────▼───────────────────────────────────────▼──────────┐
          │              MEDIAPIPE HAND LANDMARK EXTRACTION              │
          │                                                              │
          │  For each video:                                            │
          │  1. Read frames with OpenCV                                 │
          │  2. Detect hand with MediaPipe (21 landmarks per frame)     │
          │  3. Flatten: 21 × (x,y,z) = 63 features per frame          │
          │  4. Normalize to SEQUENCE_LENGTH=30 frames:                 │
          │     - Long videos → uniform sample 30 frames                │
          │     - Short videos → zero-pad to 30 frames                  │
          │  5. Skip videos with <20% hand detection                    │
          │                                                              │
          │  Output per video: numpy array of shape (30, 63)            │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                    CACHE AS .npz                              │
          │  asl_word_sequences.npz  /  arsl_word_sequences.npz          │
          │  Contains: X = (N, 30, 63), y = (N,) word_ids                │
          │  ✅ Second run loads instantly from cache                     │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                   PREPROCESSING                               │
          │                                                              │
          │  1. StandardScaler: normalize per-feature mean≈0, std≈1      │
          │  2. LabelEncoder: word_id → class index (0..N-1)             │
          │  3. One-hot encode targets                                   │
          │  4. Stratified split: 60% train / 20% val / 20% test        │
          │  5. Compute balanced class weights (handles imbalance)       │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                tf.data PIPELINE (GPU-Optimized)               │
          │                                                              │
          │  train_ds = Dataset.from_tensor_slices((X_train, y_train))   │
          │            .shuffle(buffer=min(N, 10000))                    │
          │            .batch(BATCH_SIZE)    ← 64 on GPU, 32 on CPU     │
          │            .prefetch(AUTOTUNE)   ← GPU never waits for data  │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │              BiLSTM MODEL (cuDNN-Accelerated)                 │
          │                                                              │
          │  Input(30, 63)                                               │
          │    → Bi(LSTM(128)) → BN → Drop(0.3)    [reads fwd + bkwd]   │
          │    → LSTM(64)      → BN → Drop(0.3)    [final state]        │
          │    → Dense(128, relu, L2) → Drop(0.2)  [classifier]         │
          │    → Dense(num_classes, softmax)         [prediction]        │
          │                                                              │
          │  Train: up to 100 epochs, EarlyStopping patience=15          │
          └──────────────────────────┬───────────────────────────────────┘
                                     │
          ┌──────────────────────────▼───────────────────────────────────┐
          │                     OUTPUT FILES                              │
          │                                                              │
          │  *_word_lstm_model_best.h5   ← best val_accuracy checkpoint  │
          │  *_word_lstm_model_final.h5  ← final model                   │
          │  *_word_classes.csv          ← class_index → word_id map     │
          └──────────────────────────────────────────────────────────────┘
```

---

## MediaPipe Landmark Details

```
Hand has 21 landmarks:
 0: WRIST
 1: THUMB_CMC       5: INDEX_FINGER_MCP    9: MIDDLE_FINGER_MCP   13: RING_FINGER_MCP    17: PINKY_MCP
 2: THUMB_MCP       6: INDEX_FINGER_PIP   10: MIDDLE_FINGER_PIP   14: RING_FINGER_PIP    18: PINKY_PIP
 3: THUMB_IP        7: INDEX_FINGER_DIP   11: MIDDLE_FINGER_DIP   15: RING_FINGER_DIP    19: PINKY_DIP
 4: THUMB_TIP       8: INDEX_FINGER_TIP   12: MIDDLE_FINGER_TIP   16: RING_FINGER_TIP    20: PINKY_TIP

Each landmark → (x, y, z) = 3 values
Total per frame: 21 × 3 = 63 features
Sequence per video: 30 frames × 63 features = (30, 63) tensor
```

---

## Bilingual Word Mapping

The shared vocabulary CSV creates a bridge between both language models:

```
shared_word_vocabulary.csv:
┌─────────┬──────────┬──────────┬─────────────┬─────────────┬──────────┐
│ word_id  │ english  │ arabic   │ wlasl_class │ karsl_class │ category │
├─────────┼──────────┼──────────┼─────────────┼─────────────┼──────────┤
│    0    │  drink   │  يشرب    │      1      │     161     │   verb   │
│    1    │  chair   │  كرسي    │      4      │     328     │  object  │
│    2    │  help    │  يساعد   │     12      │     182     │   verb   │
│   ...   │   ...    │   ...    │    ...      │    ...      │   ...    │
│   156   │  forgive │  يغفر   │    1753     │     446     │ religion │
└─────────┴──────────┴──────────┴─────────────┴─────────────┴──────────┘

157 words across 9 categories:
  verb (36), family (19), adjective (22), object (24),
  health (18), direction (16), job (10), social (7), religion (10)
```

Both ASL and ArSL models map to the SAME `word_id`, enabling:
- English sign → word_id=0 → show "drink / يشرب"
- Arabic sign → word_id=0 → show "drink / يشرب"
- Cross-language translation via shared ID

---

## File Structure

```
Words/
├── ASL Word (English)/          ← English ASL word training
│   ├── ASL_Word_Training.ipynb  ← main notebook (13 cells)
│   ├── asl_word_sequences.npz   ← cached data (generated)
│   ├── asl_word_lstm_model_best.h5   ← (generated)
│   ├── asl_word_lstm_model_final.h5  ← (generated)
│   └── asl_word_classes.csv     ← (generated)
│
├── ArSL Word (Arabic)/          ← Arabic ArSL word training
│   ├── ArSL_Word_Training.ipynb ← main notebook (13 cells)
│   ├── arsl_word_sequences.npz  ← cached data (generated)
│   ├── arsl_word_lstm_model_best.h5  ← (generated)
│   ├── arsl_word_lstm_model_final.h5 ← (generated)
│   └── arsl_word_classes.csv    ← (generated)
│
├── Shared/
│   └── shared_word_vocabulary.csv  ← 157 bilingual word mappings
│
├── Datasets/
│   ├── WLASL_videos/            ← 11,980 .mp4 files (✅ extracted)
│   └── KArSL_502/               ← (❌ needs download)
│
└── Docs/                        ← this documentation
    ├── MODEL_SUMMARY.md
    ├── ARCHITECTURE_AND_PIPELINE.md
    ├── DATASET_GUIDE.md
    └── TEAM_QUICKSTART.md
```
