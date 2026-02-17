# Words Module — Model Summary & Technical Specification

> **Last Updated:** February 2026  
> **Module:** `SLR Main/Words/`  
> **Team Reference Document** — Read this before running any word notebook

---

## 1. System Overview

The **Words Module** recognizes **whole sign language words** (not individual letters) from video sequences.  
It supports **two languages** using a shared bilingual vocabulary:

| Language | Dataset | Notebook | Status |
|---|---|---|---|
| **English (ASL)** | WLASL (11,980 videos) | `ASL Word (English)/ASL_Word_Training.ipynb` | ✅ Ready to train |
| **Arabic (ArSL)** | KArSL-502 | `ArSL Word (Arabic)/ArSL_Word_Training.ipynb` | ⏳ Needs KArSL download |

Both models output a **shared `word_id`** (0–156), enabling bilingual translation.

---

## 2. Model Architecture

### BiLSTM (Bidirectional Long Short-Term Memory)

```
Input: Video → 30 frames → MediaPipe → (30, 63) tensor
                                          ↓
┌─────────────────────────────────────────────────────────────┐
│  INPUT LAYER           shape = (30, 63)                     │
│    30 time steps × 63 features (21 landmarks × 3 coords)   │
├─────────────────────────────────────────────────────────────┤
│  BIDIRECTIONAL LSTM    128 units (→ 256 output)             │
│    Reads sequence forward AND backward                      │
│    return_sequences=True                                    │
│    cuDNN-accelerated (no recurrent_dropout)                 │
├─────────────────────────────────────────────────────────────┤
│  BATCH NORMALIZATION                                        │
│  DROPOUT               0.3                                  │
├─────────────────────────────────────────────────────────────┤
│  LSTM                  64 units                              │
│    Outputs final hidden state                               │
│    cuDNN-accelerated                                        │
├─────────────────────────────────────────────────────────────┤
│  BATCH NORMALIZATION                                        │
│  DROPOUT               0.3                                  │
├─────────────────────────────────────────────────────────────┤
│  DENSE                 128 units, ReLU                      │
│    he_normal init, L2 regularization (1e-4)                 │
│  DROPOUT               0.2                                  │
├─────────────────────────────────────────────────────────────┤
│  DENSE (OUTPUT)        num_classes units, Softmax           │
│    dtype=float32 (stable with mixed precision)              │
└─────────────────────────────────────────────────────────────┘
                          ↓
            Output: predicted word_id (0–156)
```

### Why BiLSTM Instead of MLP?

| | Letters (MLP) | Words (BiLSTM) |
|---|---|---|
| **Input** | Single image → (1, 63) flat | 30-frame video → (30, 63) sequence |
| **Model** | Dense layers only | Temporal layers (LSTM reads over time) |
| **What it learns** | Static hand shape | Hand shape **changes over time** |
| **Example** | 🤚 = letter "B" | 🤚→✊→👉 = word "help" |

---

## 3. Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | Frames per sample (pad short / sample long) |
| `NUM_FEATURES` | 63 | 21 MediaPipe hand landmarks × 3 (x, y, z) |
| `BATCH_SIZE` | 64 (GPU) / 32 (CPU) | Auto-selected based on hardware |
| `EPOCHS` | 100 max | EarlyStopping will stop sooner |
| `LEARNING_RATE` | 1e-3 | Reduced by ReduceLROnPlateau (×0.5 every 5 stale epochs) |
| `LSTM_UNITS_1` | 128 | BiLSTM layer (outputs 256 due to bidirectional) |
| `LSTM_UNITS_2` | 64 | Second LSTM layer |
| `DENSE_UNITS` | 128 | Classifier hidden layer |
| `DROPOUT_RATE` | 0.3 | Between LSTM layers (0.2 before output) |
| `TEST_SIZE` | 0.4 | → 60% train / 20% val / 20% test |
| `OPTIMIZER` | legacy.Adam | GPU/mixed precision compatible |

---

## 4. GPU Optimizations

| Optimization | What It Does |
|---|---|
| **Memory Growth** | `set_memory_growth(True)` — prevents TF from grabbing all VRAM |
| **cuDNN LSTM** | No `recurrent_dropout` → enables NVIDIA cuDNN kernels (5-10× faster) |
| **tf.data Pipeline** | `shuffle → batch → prefetch(AUTOTUNE)` — GPU never waits for data |
| **legacy.Adam** | Compatible with mixed precision + GPU placement |
| **clear_session()** | Cleans GPU memory before building model |
| **tf.device(DEVICE)** | Forces model + training onto GPU |
| **L2 + He Init** | Better convergence = fewer epochs needed |
| **Class Weights** | Balanced weighting for imbalanced classes |

---

## 5. Callbacks

| Callback | Config | Purpose |
|---|---|---|
| `ModelCheckpoint` | `monitor='val_accuracy', save_best_only=True` | Saves best model to `*_best.h5` |
| `EarlyStopping` | `monitor='val_loss', patience=15` | Stops training when no improvement |
| `ReduceLROnPlateau` | `factor=0.5, patience=5, min_lr=1e-7` | Halves LR when plateauing |

---

## 6. Output Artifacts

Each notebook produces:

| File | Description |
|---|---|
| `*_word_sequences.npz` | Cached extracted sequences (X, y arrays) — skip re-extraction |
| `*_word_lstm_model_best.h5` | Best checkpoint by val_accuracy |
| `*_word_lstm_model_final.h5` | Final model after early stopping |
| `*_word_classes.csv` | Maps model class index → word_id |

---

## 7. Evaluation Metrics

Both notebooks compute:
- **Top-1 Accuracy** — exact match
- **Top-5 Accuracy** — correct class in top 5 predictions
- **Training curves** — accuracy + loss over epochs
- **Confusion matrix** — heatmap of class predictions
- **Classification report** — precision, recall, F1 per class
- **Per-category accuracy** — verb, family, adjective, etc.

---

## 8. Estimated Training Times

| Phase | GPU (RTX-class) | CPU |
|---|---|---|
| MediaPipe extraction (first run) | 30–60 min | 2–4 hours |
| MediaPipe extraction (cached) | Skipped | Skipped |
| Model training (100 epochs max) | 15–30 min | 2–4 hours |
| Evaluation | < 1 min | 2–5 min |

---

## 9. Parameter Count (Approximate)

```
BiLSTM Layer 1:  ~200K params  (128 units × bidirectional)
LSTM Layer 2:    ~80K params   (64 units)
Dense Layer:     ~16K params   (128 units)
Output Layer:    ~20K params   (157 classes)
─────────────────────────────────
Total:           ~320K trainable parameters
```

This is intentionally lightweight for real-time inference on edge devices.
