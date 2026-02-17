# Team Quick-Start Guide — Words Module

> **Read time:** 5 minutes  
> **Goal:** Get any team member running the word training notebooks

---

## Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.9.x | `python --version` |
| TensorFlow | 2.10.0 | `python -c "import tensorflow; print(tensorflow.__version__)"` |
| CUDA / cuDNN | Compatible with TF 2.10 | `nvidia-smi` |
| MediaPipe | 0.10.x | `python -c "import mediapipe; print(mediapipe.__version__)"` |
| pandas | 2.0.x | `python -c "import pandas; print(pandas.__version__)"` |

### Required Python Packages
All should already be installed:
```
tensorflow==2.10.0, numpy, pandas==2.0.3, opencv-python, mediapipe,
scikit-learn, matplotlib, seaborn, tqdm, yt-dlp
```

---

## Step-by-Step: Run ASL Word Training (English)

### 1. Open the notebook
```
SLR Main/Words/ASL Word (English)/ASL_Word_Training.ipynb
```

### 2. Select Python 3.9 kernel
- Top-right of notebook → click kernel selector → choose Python 3.9.13

### 3. Run all cells in order (Cell 1 → Cell 13)

| Cell # | Name | Time | What to Watch For |
|---|---|---|---|
| 1 | Imports | ~30s | All versions printed, no import errors |
| 2 | GPU Config | ~5s | Should show `✅ GPU configured` and `Using device: /GPU:0` |
| 3 | Config | instant | All paths show ✅ |
| 4 | Load Vocab | ~2s | `157 words`, download candidates count |
| 5 | Download | instant | `RUN_DOWNLOAD=False` — skip (videos already extracted) |
| 6 | MediaPipe Extract | **30-60 min (first time)** / instant (cached) | Progress bar; creates `.npz` |
| 7 | Data Exploration | ~5s | Bar chart + heatmap; check class distribution |
| 8 | Preprocessing | ~3s | Shows 60/20/20 split sizes |
| 9 | Train BiLSTM | **15-30 min** | Watch val_accuracy climb; cuDNN LSTM enabled |
| 10 | Evaluation | ~1 min | Top-1 accuracy, confusion matrix, per-category |

### 4. Output files
After training, you'll find in `ASL Word (English)/`:
- `asl_word_lstm_model_best.h5` — use this for inference
- `asl_word_lstm_model_final.h5` — backup
- `asl_word_classes.csv` — class index → word_id mapping

---

## Step-by-Step: Run ArSL Word Training (Arabic)

### 0. Download KArSL first!
- Go to: https://www.kaggle.com/datasets/yousefelkilany/karsl-502
- Download and extract to: `SLR Main/Words/Datasets/KArSL_502/`

### 1. Open the notebook
```
SLR Main/Words/ArSL Word (Arabic)/ArSL_Word_Training.ipynb
```

### 2-4. Same as ASL (select kernel → run all → check outputs)

---

## Common Issues & Fixes

| Problem | Symptom | Fix |
|---|---|---|
| **No GPU** | Cell 2 shows "No GPU detected" | Check CUDA install, run `nvidia-smi` |
| **Out of Memory** | `OOM when allocating tensor` | Reduce `BATCH_SIZE` to 16 or 8 in Config cell |
| **NaN loss** | Loss shows `nan` during training | Set `ENABLE_MIXED_PRECISION = False` in GPU cell |
| **Slow LSTM** | Training takes hours on GPU | Check cuDNN — `recurrent_dropout` should NOT be set |
| **Import error** | `ModuleNotFoundError` | `pip install <missing_package>` |
| **KArSL not found** | Cell 8 shows `FileNotFoundError` | Download KArSL from Kaggle (see above) |
| **Low accuracy** | Top-1 < 30% | Normal for 157 classes — increase epochs or add data |
| **Extraction crashed** | MediaPipe stopped mid-video | Re-run cell 6 — it'll resume from cache |

---

## Folder Structure Overview

```
SLR Main/
├── Letters/                    ← EXISTING: letter-level recognition
│   ├── ASL Letter (English)/   ← MLP on single frames
│   ├── ArSL Letter (Arabic)/   ← MLP on single frames
│   ├── Datasets/
│   └── Guides/
│
├── Words/                      ← NEW: word-level recognition
│   ├── ASL Word (English)/     ← BiLSTM on 30-frame sequences
│   ├── ArSL Word (Arabic)/     ← BiLSTM on 30-frame sequences
│   ├── Shared/                 ← shared_word_vocabulary.csv
│   ├── Datasets/               ← WLASL_videos/ + KArSL_502/
│   └── Docs/                   ← you are here
```

---

## Key Contacts & Resources

| Resource | Location |
|---|---|
| Model Summary | `Words/Docs/MODEL_SUMMARY.md` |
| Architecture Diagram | `Words/Docs/ARCHITECTURE_AND_PIPELINE.md` |
| Dataset Details | `Words/Docs/DATASET_GUIDE.md` |
| Bilingual Vocabulary | `Words/Shared/shared_word_vocabulary.csv` |
| Letter Guides | `Letters/Guides/` |

---

## Future: Letters + Words Combined System

See `Words/Docs/LETTERS_WORDS_INTEGRATION.md` for how the letter and word models
will work together in a unified real-time system.
