# SLR Project — Complete Guide

Everything about the Sign Language Recognition system: models, architecture, notebooks, and how to run.

---

## Project Overview

This project recognizes **sign language** in two languages and two modes:

| Model | Language | Input | Classes | Architecture | Status |
|-------|----------|-------|---------|-------------|--------|
| ASL Letters | English | Single frame (63 features) | 29 (A-Z + del/nothing/space) | MLP | Trained |
| ArSL Letters | Arabic | Single frame (63 features) | 31 (28 Arabic letters + del/nothing/space) | MLP | Trained |
| ASL Words | English | 30-frame sequence (30x63) | 157 words | BiLSTM | Training notebook ready |
| ArSL Words | Arabic | 30-frame sequence (30x63) | 157 words | BiLSTM | Training notebook ready |

---

## How It Works

```
Webcam → MediaPipe Hands → 21 landmarks × 3 coords = 63 features
                                    |
                    ┌───────────────┴───────────────┐
                    │                               │
             Still hand?                     Moving hand?
                    │                               │
            LETTER MODEL (MLP)            WORD MODEL (BiLSTM)
            Single frame (1,63)           30 frames (30,63)
            → A, B, C, ...               → hello, drink, ...
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                            Sentence Builder
                            → "my name AHMED help"
```

### Letter Recognition (Single Frame)
- MediaPipe extracts 21 hand landmarks from ONE frame
- Flattened to 63 features (21 × 3 coordinates)
- Fed into MLP → predicts letter
- Uses "commit-once-then-wait" to avoid repetition

### Word Recognition (30-Frame Sequence)
- MediaPipe extracts landmarks from 30 consecutive frames
- Stacked into (30, 63) sequence
- Fed into BiLSTM → predicts word
- Uses cooldown-based commitment

---

## MLP Architecture (Letters)

Used for both English and Arabic letters. ~23K parameters.

```
Input(63)
  → Dense(256, ReLU, L2=1e-4) → BatchNorm → Dropout(0.3)
  → Dense(128, ReLU, L2=1e-4) → BatchNorm → Dropout(0.25)
  → Dense(64, ReLU) → Dropout(0.2)
  → Dense(num_classes, Softmax)
```

Training: Adam(lr=0.001), 20 epochs max, EarlyStopping(patience=5)

## BiLSTM Architecture (Words)

Used for both English and Arabic words. ~320K parameters.

```
Input(30, 63)
  → Bidirectional(LSTM(128, return_sequences=True)) → BatchNorm → Dropout(0.3)
  → Bidirectional(LSTM(64)) → BatchNorm → Dropout(0.3)
  → Dense(128, ReLU) → Dropout(0.2)
  → Dense(num_classes, Softmax)
```

Training: Adam(lr=0.001), 50 epochs max, EarlyStopping(patience=7)

---

## Folder Structure & Notebook Functions

```
Letters_ORIGINAL/
│
├── Base_Pipeline_English_Letters/        ← ENGLISH LETTER FILES
│   ├── Mediapipe_Training.ipynb          MLP training + webcam inference
│   ├── MobileNetV2_Training.ipynb        MobileNetV2 training (optional)
│   ├── Combined_Architecture.ipynb       MLP + MobileNet fusion inference
│   ├── Production_Architecture_English.ipynb  Production webcam loop
│   ├── SLR_Diagnostics.ipynb             Model analysis & debugging
│   ├── asl_mediapipe_keypoints_dataset.csv   Training data (63 features + label)
│   ├── asl_mediapipe_mlp_model.h5        Trained MLP model
│   └── sign_language_model_MobileNetV2.h5  Trained MobileNet model
│
├── ArSL (Arabic Letters)/                ← ARABIC LETTER FILES
│   ├── Mediapipe_Final_Arabic1.ipynb     MLP training + webcam inference
│   ├── Combined_Architecture_Arabic_GPU.ipynb  MLP + MobileNet fusion
│   ├── Mobilenet_Arabic_Best_Final.ipynb MobileNetV2 training
│   ├── Production_Architecture_Arabic.ipynb  Production webcam loop
│   ├── FINAL_CLEAN_DATASET.csv           Training data (63 features + label)
│   ├── arsl_mediapipe_mlp_model_final.h5 Trained MLP model
│   └── mobilenet_arabic_final.h5         Trained MobileNet model
│
├── ASL_Word_Training.ipynb               ← ASL WORD PIPELINE
│   Downloads WLASL videos from Kaggle, extracts MediaPipe landmarks,
│   builds 30-frame sequences, trains BiLSTM.
│   Output: asl_word_sequences.npz + asl_word_lstm_model_best.h5
│
├── ArSL_Word_Training.ipynb              ← ARSL WORD PIPELINE
│   Downloads KArSL-502 from Kaggle, extracts MediaPipe landmarks,
│   builds 30-frame sequences, trains BiLSTM.
│   Output: arsl_word_sequences.npz + arsl_word_lstm_model_best.h5
│
├── Unified_Dataset_Merger.ipynb          ← DATASET MERGER
│   Merges, balances, and splits datasets for all 4 models.
│   Run AFTER the word training notebooks produce their NPZ files.
│
├── shared_word_vocabulary.csv            ← 157 bilingual word mappings
│   For future English↔Arabic translation feature.
│
├── Arabic guide/                         ← HELPER SCRIPTS
│   ├── arabic_class_labels.py            Class label definitions
│   ├── arabic_data_collection.py         Webcam data collection tool
│   └── arabic_display_utils.py           RTL text rendering
│
├── Docs/                                 ← DOCUMENTATION
├── Guides/                               ← Reference implementations
└── Orignal Notebooks/                    ← Backup copies
```

### What Each Notebook Does

| Notebook | Purpose | Run When |
|----------|---------|----------|
| `Mediapipe_Training.ipynb` | Trains English MLP on CSV landmarks + live webcam test | To retrain English letter model |
| `MobileNetV2_Training.ipynb` | Trains English MobileNet on raw images | Optional enhancement |
| `Combined_Architecture.ipynb` | Runs both MLP + MobileNet together, fuses predictions | For best English accuracy |
| `Production_Architecture_English.ipynb` | Production webcam loop (English) | Live demo |
| `Mediapipe_Final_Arabic1.ipynb` | Trains Arabic MLP on CSV landmarks + live webcam test | To retrain Arabic letter model |
| `Mobilenet_Arabic_Best_Final.ipynb` | Trains Arabic MobileNet | Optional enhancement |
| `Combined_Architecture_Arabic_GPU.ipynb` | MLP + MobileNet fusion (Arabic) | For best Arabic accuracy |
| `Production_Architecture_Arabic.ipynb` | Production webcam loop (Arabic) | Live demo |
| `ASL_Word_Training.ipynb` | Full ASL word pipeline (download→extract→train) | To build ASL word model |
| `ArSL_Word_Training.ipynb` | Full ArSL word pipeline (download→extract→train) | To build ArSL word model |
| `Unified_Dataset_Merger.ipynb` | Merge + balance + split all datasets | After word NPZ files exist |
| `SLR_Diagnostics.ipynb` | Debug & analyze model performance | When troubleshooting |

---

## Class Labels

### English ASL (29 classes)
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, del, nothing, space

### Arabic ArSL (31 classes)
ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي + space, del, nothing

---

## How to Run

### Letters (already trained)
1. Open `Combined_Architecture.ipynb` (English) or `Combined_Architecture_Arabic_GPU.ipynb` (Arabic)
2. Run all cells → webcam opens with live recognition

### Words (need to train first)
1. Run `ASL_Word_Training.ipynb` — downloads ~5.4GB from Kaggle, processes videos, trains BiLSTM
2. Run `ArSL_Word_Training.ipynb` — downloads KArSL-502, same pipeline
3. Run `Unified_Dataset_Merger.ipynb` — merges and balances all datasets

### Cell 2 in ASL_Word_Training.ipynb
- **Downloads ~5.4 GB** to Kaggle's cache folder (`~/.cache/kagglehub/`)
- Contains 12K pre-downloaded videos + JSON metadata
- **You do NOT need to re-run it** — kagglehub caches the download. Running it again just reuses the cache.

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Unflipped frames for training | MediaPipe landmarks differ on flipped vs unflipped frames |
| No hand mirroring | Training data was collected without mirroring |
| Commit-once-then-wait | Prevents "mmmmmooooccc" letter repetition |
| Wrist-relative normalization (words) | Translation invariance for video sequences |
| Forward-fill missing frames | Maintains sequence continuity when MediaPipe misses a frame |
| GPU with mixed_float16 | Faster training, automatic CPU fallback |
| Signer-aware splitting | Prevents memorizing specific signers' hand shapes |

---

## Trained Model Files

| Model | File | Size | Classes |
|-------|------|------|---------|
| English MLP | `asl_mediapipe_mlp_model.h5` | ~300 KB | 29 |
| Arabic MLP | `arsl_mediapipe_mlp_model_final.h5` | ~310 KB | 31 |
| English MobileNet | `sign_language_model_MobileNetV2.h5` | ~9 MB | 29 |
| Arabic MobileNet | `mobilenet_arabic_final.h5` | ~18 MB | 34 |
| English BiLSTM | `asl_word_lstm_model_best.h5` | ~1.5 MB | 157 (after training) |
| Arabic BiLSTM | `arsl_word_lstm_model_best.h5` | ~1.5 MB | 157 (after training) |
