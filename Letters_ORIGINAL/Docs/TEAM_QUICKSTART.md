# Team Quick-Start Guide — Letters Module

> **Read time:** 5 minutes  
> **Goal:** Get any team member running the letter recognition notebooks

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9.x |
| TensorFlow | 2.10+ |
| CUDA/cuDNN | Compatible with your TF version (optional) |
| MediaPipe | 0.10.x |
| OpenCV | 4.x |
| pandas | 2.0.x |
| scikit-learn | 1.x |

**Install all dependencies:**
```bash
pip install -r "Letters/ASL Letter (English)/requirements.txt"
```

Or manually:
```bash
pip install tensorflow mediapipe opencv-python numpy pandas scikit-learn matplotlib pillow arabic-reshaper python-bidi
```

---

## Quick Start: English ASL Letters

### Option A: Combined Notebook (Recommended)

**File:** `Letters/ASL Letter (English)/Combined_Architecture.ipynb`

| Cell | Name | Time | What to Watch For |
|---|---|---|---|
| 1–6 | Imports + Config | ~30s | Library versions printed, camera URL set |
| 7 | Settings | instant | Stabilization parameters defined |
| 8–12 | Model Loading | ~10s | MLP + MobileNet loaded (MobileNet optional) |
| **13** | **Camera Loop (Combined)** | **Interactive** | **Press 'q' to quit, 'c' to clear** |
| 14–24 | Fine-tuning (optional) | 5–15 min | Retrains MLP with augmentation |
| **25** | **Camera Loop (Fine-tuned)** | **Interactive** | MLP-only inference |

**Run Cells 1–13** to start the camera with the pre-trained model.

### Option B: MediaPipe Training Notebook

**File:** `Letters/ASL Letter (English)/Mediapipe_Training.ipynb`

| Cell | Name | Time | What to Watch For |
|---|---|---|---|
| 1–3 | Imports + GPU | ~10s | GPU detected (or CPU fallback) |
| 4 | Load CSV | ~2s | 29 classes loaded |
| 5–8 | Preprocessing | ~5s | Train/val/test split |
| 9–11 | Build + Train MLP | 2–5 min | Watch val_accuracy |
| 12–17 | Evaluation | ~30s | Confusion matrix |
| **18** | **Camera Loop** | **Interactive** | Real-time recognition |

---

## Quick Start: Arabic ArSL Letters

### Option A: Final Combined Notebook

**File:** `Letters/ArSL Letter (Arabic)/Final Notebooks/Combined_Architecture_Arabic_GPU.ipynb`

| Cell | Name | Time | What to Watch For |
|---|---|---|---|
| 1–3 | Imports + GPU Config | ~10s | float32 policy set |
| 4–13 | Markdown + Setup | instant | Architecture docs |
| 14 | Load Models | ~10s | MLP loaded, MobileNet may be disabled |
| 15–23 | Config + Helpers | instant | Settings defined |
| **24** | **Camera Loop** | **Interactive** | Arabic letter recognition |

### Option B: MediaPipe Training Notebook

**File:** `Letters/ArSL Letter (Arabic)/Mediapipe_Final_Arabic.ipynb`

Run all cells in order. Camera loop is in the last code cell.

### Option C: Standalone Arabic MLP

**File:** `Letters/ArSL Letter (Arabic)/Final Notebooks/Mediapipe_Final_Arabic1.ipynb`

Run all cells. Camera loop is Cell 18.

---

## Camera Controls (All Notebooks)

| Key | Action |
|---|---|
| `q` | Quit camera |
| `c` | Clear sentence |
| `s` | Save screenshot (some notebooks) |

**Window is resizable** — drag corners to resize.

---

## How Inference Works

1. **Show your hand** to the camera
2. **Hold a sign** steady — you'll see "Stabilizing..." then "Hold: XX%"
3. Once the progress bar reaches 100%, the letter **commits** (turns green)
4. **Change to a different sign** or **remove your hand** to spell the next letter
5. Same sign stays locked until you change — prevents "mmmmmooo" repetition

---

## Common Issues & Fixes

| Problem | Fix |
|---|---|
| **Camera won't open** | Try changing `CAMERA_INDEX` (0→1→2), or set `CAMERA_SOURCE` |
| **Model crash on hand detection** | Ensure `tf.keras.mixed_precision.set_global_policy('float32')` is called before loading (already fixed in active notebooks) |
| **Wrong predictions** | Check that frame is NOT flipped before MediaPipe. Training used unflipped frames. |
| **Letter repeats (mmmooo)** | Use the fixed notebooks with commit-once-then-wait strategy |
| **No GPU detected** | Run `nvidia-smi` to check CUDA. Notebooks work fine on CPU. |
| **Import error** | `pip install <missing package>` |
| **Arabic text not displaying** | Install `arabic-reshaper` and `python-bidi` |
| **MobileNet disabled** | Normal — class count mismatch causes auto-disable. MLP-only works well. |
| **Low accuracy** | Ensure good lighting, hand centered in frame, single hand only |
| **Tkinter / display error** | Running on server? Need a display. Use `Xvfb` or run locally. |

---

## Notebook Map

```
Letters/
├── ASL Letter (English)/
│   ├── Combined_Architecture.ipynb    ← ⭐ PRIMARY (MLP + MobileNet)
│   ├── Mediapipe_Training.ipynb       ← MLP training + camera
│   ├── MobileNetV2_Training.ipynb     ← MobileNet training only
│   ├── mediapipe_draft.ipynb          ← exploration (no camera)
│   └── MBV2_draft.ipynb              ← draft (no camera)
│
├── ArSL Letter (Arabic)/
│   ├── Mediapipe_Final_Arabic.ipynb   ← ⭐ PRIMARY Arabic MLP
│   ├── Final Notebooks/
│   │   ├── Combined_Architecture_Arabic_GPU.ipynb  ← ⭐ Arabic Fusion
│   │   ├── Mediapipe_Final_Arabic1.ipynb           ← Arabic MLP (copy)
│   │   └── Mobilenet_Arabic_Best_Final.ipynb       ← Arabic MobileNet
│   ├── Mediapipe_Training.ipynb       ← older training version
│   └── [other experimental notebooks]
│
├── Orignal Notebooks/                 ← ⚠️ BACKUP ONLY (unfixed)
│   └── [original copies before fixes]
│
└── Docs/                              ← ← you are here
```

**⭐ = Recommended notebooks to use**  
**⚠️ = Orignal Notebooks are unfixed backups — do not use for inference**

---

## Model Files Reference

| File | Language | Type | Classes |
|---|---|---|---|
| `asl_mediapipe_mlp_model.h5` | English | MLP | 29 |
| `asl_mediapipe_mlp_model_best.h5` | English | MLP | 29 |
| `best_model_finetuned.h5` | English | Fine-tuned MLP | 29 |
| `sign_language_model_MobileNetV2.h5` | English | MobileNet | 29 |
| `arsl_mediapipe_mlp_model_best.h5` | Arabic | MLP | 31 |
| `arsl_mediapipe_mlp_model_final.h5` | Arabic | MLP | 34 |
| `mobilenet_arabic_final.h5` | Arabic | MobileNet | 35 |

---

## For More Details

- [MODEL_SUMMARY.md](MODEL_SUMMARY.md) — Model architecture & hyperparameters
- [ARCHITECTURE_AND_PIPELINE.md](ARCHITECTURE_AND_PIPELINE.md) — Full data flow diagrams
- [DATASET_GUIDE.md](DATASET_GUIDE.md) — Dataset download & structure
- [CLASS_LABELS.md](CLASS_LABELS.md) — All letter labels (English & Arabic)
- [LETTERS_WORDS_INTEGRATION.md](LETTERS_WORDS_INTEGRATION.md) — How letters + words combine
