# Letters Module — Model Summary & Technical Specification

> **Last Updated:** February 2026  
> **Module:** `SLR Main/Letters/`

---

## 1. System Overview

The Letters Module recognizes **individual sign language letters** from single video frames. Two languages supported:

| Language | Dataset | Notebooks | Classes | Status |
|---|---|---|---|---|
| **English (ASL)** | Kaggle ASL Alphabet (87,000 images) | `Combined_Architecture.ipynb`, `Mediapipe_Training.ipynb` | 29 | ✅ Trained |
| **Arabic (ArSL)** | Custom Arabic dataset | `Mediapipe_Final_Arabic.ipynb`, `Combined_Architecture_Arabic_GPU.ipynb` | 31–34 | ✅ Trained |

Both languages use the same MLP architecture for MediaPipe landmark-based inference. MobileNetV2 is available as an optional image-based model for fusion.

---

## 2. Model Architectures

### 2a. MLP (MediaPipe Landmarks) — Primary Model

Used for both English and Arabic. Trained on **63 features** (21 hand landmarks × 3 coordinates).

```
Input(63)
  → Dense(256, ReLU, he_normal, L2=1e-4)
  → BatchNormalization → Dropout(0.3)
  → Dense(128, ReLU, he_normal, L2=1e-4)
  → BatchNormalization → Dropout(0.25)
  → Dense(64, ReLU, he_normal)
  → Dropout(0.2)
  → Dense(num_classes, Softmax, float32)
```

| Variant | Output Classes | Model File |
|---|---|---|
| English ASL | 29 (A–Z + del, nothing, space) | `asl_mediapipe_mlp_model.h5` / `best.h5` |
| Arabic ArSL | 31 (28 letters + space, del, nothing) | `arsl_mediapipe_mlp_model_best.h5` / `final.h5` |

**Why MLP?** Letters are **static hand poses** — a single frame contains all the information needed. No temporal sequence required (unlike words which need BiLSTM on 30 frames).

### 2b. MobileNetV2 (Image-Based) — Optional Fusion Model

Used as an alternative/fusion model in Combined notebooks. Trained on **hand crop images**.

| Variant | Input Size | Classes | Model File |
|---|---|---|---|
| English ASL | 128 × 128 × 3 | 29 | `sign_language_model_MobileNetV2.h5` |
| Arabic ArSL | 96 × 96 × 3 | 34–35 | `mobilenet_arabic_final.h5` |

**Note:** In the Arabic Combined notebook, MobileNet is automatically disabled when its class count doesn't match the MLP's. The system falls back to MLP-only inference.

---

## 3. Training Hyperparameters

| Parameter | Value |
|---|---|
| Input Features | 63 (21 landmarks × 3 coords) |
| Optimizer | `legacy.Adam(lr=0.001)` |
| Loss | `categorical_crossentropy` |
| Epochs | 20 max (with EarlyStopping) |
| Batch Size | 256 (GPU) / 64 (CPU) |
| Mixed Precision | `mixed_float16` on GPU, `float32` on CPU |
| Train / Val / Test Split | 64% / 16% / 20% |

---

## 4. Callbacks

| Callback | Configuration |
|---|---|
| **ModelCheckpoint** | Save best by `val_accuracy` |
| **EarlyStopping** | `patience=5`, monitor `val_loss`, restore best weights |
| **ReduceLROnPlateau** | `factor=0.5`, `patience=3`, `min_lr=1e-7` |

---

## 5. GPU Optimizations

- **Memory growth** enabled to prevent TF from allocating all GPU memory
- **Mixed precision** (`mixed_float16`) for faster training on GPU
- **Legacy Adam optimizer** for compatibility with older CUDA versions
- **float32 policy** forced at load time for CPU inference (prevents crashes)

---

## 6. MediaPipe Configuration

```python
mp_hands.Hands(
    model_complexity=0,          # Fastest model (0 = lite)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2              # Detect up to 2 (but only process 1)
)
```

Each hand produces **21 landmarks** with (x, y, z) coordinates:

```
 0: WRIST
 1-4:   THUMB    (CMC, MCP, IP, TIP)
 5-8:   INDEX    (MCP, PIP, DIP, TIP)
 9-12:  MIDDLE   (MCP, PIP, DIP, TIP)
13-16:  RING     (MCP, PIP, DIP, TIP)
17-20:  PINKY    (MCP, PIP, DIP, TIP)

Total: 21 × 3 = 63 features per frame
```

---

## 7. Inference Strategy — Commit-Once-Then-Wait

All camera loops use the **commit-once-then-wait** strategy to prevent letter repetition:

1. Predictions enter a **stability buffer** (10 frames, 7/10 majority required)
2. Stable prediction must be held for **0.8 seconds** continuously
3. Once committed → system **locks** that label
4. Unlock only when: **(a)** hand leaves frame, OR **(b)** a different stable sign appears
5. This prevents "mmmmmoooocccc" repetition — each sign = exactly one letter

**Key parameters:**

| Parameter | Value | Purpose |
|---|---|---|
| `STABILIZATION_WINDOW_SIZE` | 10 | Buffer size for majority voting |
| `STABILIZATION_THRESHOLD` | 7 | Min votes needed (7/10 = 70%) |
| `MIN_CONFIDENCE` | 0.70 | Minimum model confidence to consider |
| `HOLD_TIME_REQUIRED` | 0.8s | Must hold sign this long before commit |

---

## 8. Output Artifacts

| File | Description |
|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | English training data (63 features + label) |
| `arabic_final_training_data.csv` | Arabic training data |
| `*_mlp_model.h5` / `*_best.h5` | Trained MLP model weights |
| `*_MobileNetV2.h5` | Trained MobileNet model weights |
| `best_model_finetuned.h5` | Fine-tuned MLP (English) |
| `training_history_*.csv` | Training logs |

---

## 9. Estimated Training Times

| Phase | GPU | CPU |
|---|---|---|
| Data collection (MediaPipe) | 10–30 min | 30–60 min |
| MLP training (20 epochs) | 2–5 min | 10–20 min |
| MobileNetV2 training | 15–30 min | 1–2 hours |

---

## 10. Parameter Counts

| Model | Parameters | Size |
|---|---|---|
| MLP (29 classes) | ~23K | ~300 KB |
| MLP (31 classes) | ~24K | ~310 KB |
| MobileNetV2 | ~2.3M | ~9 MB |

Both are intentionally lightweight for real-time edge inference.
