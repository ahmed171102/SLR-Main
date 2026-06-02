# SLR Project — Repository Structure & Organization Guide

> Sign Language Recognition (ASL + ArSL) — Letters & Words
> Use this map to organize files for the repo and team sharing.
>
> **Status:** Proposed structure for review. Nothing has been moved yet — the
> `data/ cache/ artifacts/ notebooks/ live/` subfolders below are recommendations.

---

## 0. Conventions

- **PROJECT_ROOT** = `M:/Term 10/Grad/` (notebooks reference this)
- **Repo root** = `M:/Term 10/Grad/SLR Main/`
- Each model lives under one **language + task** folder, organized as:
  `data/` · `cache/` · `notebooks/` · `artifacts/` · `live/`
- Large binaries (`.h5`, `.npz`, `.rar`, `.mp4`) are **git-ignored** — share via external storage / Git LFS.

---

## 1. Top-Level Layout

```
M:/Term 10/Grad/                          <- PROJECT_ROOT
|-- SLR Main/                             <- Git repo
|   |-- Letters/                          <- Letter models (MLP + MobileNetV2)
|   |-- Words/                            <- Word models (BiLSTM)
|   |-- Separated Pipelines/              <- Production inference (FastAPI)
|   |-- Deployment/docs/                  <- Deployment guides
|   |-- scripts/                          <- Helper / patch scripts
|   |-- frontend/                         <- Web app (senior project)
|   |-- Current Thesis/                   <- Thesis (LaTeX + figures)
|   |-- Plans/                            <- Diagrams, research
|   |-- Testing repos/                    <- External reference repos (not core)
|   `-- README.md
`-- Words dataset/                        <- EXTERNAL raw WLASL data (outside repo)
    |-- WLASL_v0.3.json
    |-- nslt_*.json
    `-- Words Datasets/WLASL_videos/      <- ~11,980 .mp4 files
```

---

## 2. The Four Models (Core)

| # | Model | Type | Input | Classes | Canonical Output |
|---|-------|------|-------|---------|------------------|
| 1 | ASL Letters | MLP + MobileNetV2 | 63 keypoints / 128x128 img | 29 | `asl_mediapipe_mlp_model.h5` |
| 2 | ArSL Letters | MLP + MobileNetV2 | 63 keypoints / 96x96 img | 31-35 | `arsl_mediapipe_mlp_model_final.h5` |
| 3 | ASL Words | BiLSTM | (30, 63) sequence | 157 (shared vocab) | `asl_word_lstm_model_best.h5` |
| 4 | ArSL Words | BiLSTM | (30, 63) sequence | 502 / subset | `arsl_v2_best.h5` |

> All models use the same MediaPipe feature: **21 hand landmarks x (x,y,z) = 63 features per frame.**

---

## 3. Model 1 — ASL Letters (English)

**Folder:** `Letters/ASL Letter (English)/`

| Role | File |
|------|------|
| Raw images (external) | `Letters/Datasets/Asl_Sign_Data/` (Kaggle ASL alphabet, ~87k imgs) |
| Keypoint dataset | `asl_mediapipe_keypoints_dataset.csv` (63 cols + label) |
| Train (MLP) | `Mediapipe_Training.ipynb` |
| Train (MobileNet) | `MobileNetV2_Training.ipynb` |
| Live / fusion | `Combined_Architecture.ipynb`, `Production_Architecture_English.ipynb` |
| Model — MLP | `asl_mediapipe_mlp_model.h5` |
| Model — MobileNet | `sign_language_model_MobileNetV2_updated.h5` |

**Recommended layout**
```
Letters/ASL Letter (English)/
|-- data/        asl_mediapipe_keypoints_dataset.csv
|-- notebooks/   Mediapipe_Training.ipynb | MobileNetV2_Training.ipynb | Combined_Architecture.ipynb
|-- artifacts/   asl_mediapipe_mlp_model.h5 | sign_language_model_MobileNetV2_updated.h5
`-- live/        Production_Architecture_English.ipynb
```
Classes: A-Z + `space`, `del`, `nothing` (29).

---

## 4. Model 2 — ArSL Letters (Arabic)

**Folder:** `Letters/ArSL Letter (Arabic)/`

| Role | File |
|------|------|
| Raw images (external) | `Letters/Datasets/Dataset (ArASL)/` (often `.rar`) |
| Training CSV | `Final Notebooks/FINAL_CLEAN_DATASET.csv` (also `arabic_final_training_data.csv`) |
| Train (MLP) | `Mediapipe_Final_Arabic.ipynb` |
| Train (MobileNet) | `Final Notebooks/Mobilenet_Arabic_Best_Final.ipynb` |
| Live | `Final Notebooks/Combined_Architecture_Arabic_GPU.ipynb` |
| Model — MLP | `Final Notebooks/arsl_mediapipe_mlp_model_final.h5` |
| Model — MobileNet | `Final Notebooks/mobilenet_arabic_final.h5` |

**Recommended layout**
```
Letters/ArSL Letter (Arabic)/
|-- data/        FINAL_CLEAN_DATASET.csv
|-- notebooks/   Mediapipe_Final_Arabic.ipynb | Mobilenet_Arabic_Best_Final.ipynb
|-- artifacts/   arsl_mediapipe_mlp_model_final.h5 | mobilenet_arabic_final.h5
`-- live/        Combined_Architecture_Arabic_GPU.ipynb
```
Classes: 28 letters + `space`, `del`, `nothing` (31 core; some models 34-35).
**Cleanup:** root-level duplicate `.h5` (`*_best.h5`, `mobilenet_arabic_best_*`) -> archive.

---

## 5. Model 3 — ASL Words (English)

**Folder:** `Words/ASL Word (English)/`

| Role | File |
|------|------|
| Raw videos (external) | `Words dataset/Words Datasets/WLASL_videos/*.mp4` |
| Metadata | `WLASL_v0.3.json`, `nslt_2000.json`, `wlasl_class_list.txt` |
| Shared vocab | `Words/Shared/shared_word_vocabulary.csv` (157 EN<->AR words) |
| Sequence cache | `asl_word_sequences.npz` -> X:(N,30,63), y:word_ids |
| Train | `ASL_Word_Training.ipynb` / `Unified_Word_Training_Version2_split.ipynb` |
| Live | `ASL_Word_Live_Test.ipynb` |
| Model | `asl_word_lstm_model_best.h5` | `..._final.h5` |
| Class map | `asl_word_classes.csv` |

**Recommended layout**
```
Words/ASL Word (English)/
|-- data/        WLASL_v0.3.json | nslt_2000.json | asl_word_vocabulary.csv
|-- cache/       asl_word_sequences.npz
|-- notebooks/   ASL_Word_Training.ipynb | Unified_Word_Training_Version2_split.ipynb
|-- artifacts/   asl_word_lstm_model_best.h5 | ..._final.h5 | asl_word_classes.csv
`-- live/        ASL_Word_Live_Test.ipynb
```
**Cleanup:** `ASL_Word_Training 1.ipynb`, `.backup*`, multiple `Unified_*` copies -> keep one, archive rest.

---

## 6. Model 4 — ArSL Words (Arabic)

**Folder:** `Words/ArSL Word (Arabic)/`

| Role | File |
|------|------|
| Raw KArSL (external) | `Words/Datasets/KArSL_502/{class_id}/` (`.npy`/`.csv`/`.mp4`) |
| Labels | `KARSL-502_Labels.csv` | `KARSL-502_BasicWords.csv` | `KARSL-502_Labels.txt` |
| Full cache | `arsl_word_sequences_v2_full.npz` (must be generated) |
| Custom subset | `arsl_custom_subset.npz` |
| Train (main) | `ArSL_Word_Training_v2.ipynb` (builds full NPZ in Cell 6) |
| Train (subset) | `ArSL_Word_Training_CustomWords.ipynb` |
| Live | `ArSL_Word_Live_Test.ipynb` |
| Model | `arsl_v2_best.h5` | `arsl_v2_final.h5` (legacy: `arsl_word_lstm_model_best.h5`) |
| Scaler / classes | `arsl_v2_scaler.npz` | `arsl_v2_classes.csv` |

**Recommended layout**
```
Words/ArSL Word (Arabic)/
|-- data/        KARSL-502_Labels.csv | KARSL-502_BasicWords.csv
|-- cache/       arsl_word_sequences_v2_full.npz | arsl_custom_subset.npz
|-- notebooks/   ArSL_Word_Training_v2.ipynb | ArSL_Word_Training_CustomWords.ipynb
|-- artifacts/   arsl_v2_best.h5 | arsl_v2_final.h5 | arsl_v2_scaler.npz | arsl_v2_classes.csv
|   `-- plots/   *_training_curves.png | *_confusion_matrix.png | *_f1_scores.png
`-- live/        ArSL_Word_Live_Test.ipynb
```
> **Blocker:** the `CustomWords` notebook fails until `arsl_word_sequences_v2_full.npz` exists.
> Run `ArSL_Word_Training_v2.ipynb` -> Cell 6 first.

---

## 7. Shared Vocabulary (links both word models)

```
Words/Shared/
|-- shared_word_vocabulary.csv   <- 157 matched EN<->AR words (word_id 0-156, 9 categories)
`-- Merge_Vocab.ipynb
```
Columns: `word_id | english | arabic | wlasl_class | karsl_class | category`.

---

## 8. Production Pipeline (FastAPI)

```
Separated Pipelines/
|-- config/settings.py        <- ALL model paths + thresholds (single source of truth)
|-- english_pipeline/         <- letter/word predictors + decoders
|-- arabic_pipeline/          <- letter/word predictors + decoders
|-- shared/
|   |-- models/               <- model_loader.py | temporal_attention.py
|   `-- utils/                <- mediapipe_extractor.py | mode_detector.py
|-- llm_agent/                <- correction_agent.py | prompts.py
|-- backend/app/              <- main.py | routes/ | schemas/
`-- tests/
```

**Paths `settings.py` expects (keep these names stable):**

| Model | Path |
|-------|------|
| ASL letter MLP | `Letters/ASL Letter (English)/asl_mediapipe_mlp_model.h5` |
| ASL letter MBV2 | `Letters/ASL Letter (English)/sign_language_model_MobileNetV2_updated.h5` |
| ArSL letter MLP | `Letters/ArSL Letter (Arabic)/Final Notebooks/arsl_mediapipe_mlp_model_final.h5` |
| ArSL letter MBV2 | `Letters/ArSL Letter (Arabic)/Final Notebooks/mobilenet_arabic_final.h5` |
| ASL word LSTM | `Words/ASL Word (English)/asl_word_lstm_model_best.h5` |
| ArSL word LSTM | `Words/ArSL Word (Arabic)/arsl_word_lstm_model_best.h5` |
| Shared vocab | `Words/Shared/shared_word_vocabulary.csv` |

> If you rename/move model files, update `Separated Pipelines/config/settings.py` accordingly.

---

## 9. Documentation Index

```
Letters/Docs/      MODEL_SUMMARY.md | DATASET_GUIDE.md
Words/Docs/        MODEL_SUMMARY.md | DATASET_GUIDE.md | ARCHITECTURE_AND_PIPELINE.md
                   ArSL_Word_Model_Report.md | Unified_Bilingual_Words_Models_Report.md
Deployment/docs/   01_INVENTORY -> 11_TIMELINE (incl. 04_FOLDER_STRUCTURE.md, 08_MODEL_CONVERSION.md)
```

---

## 10. Keep vs Archive (declutter for sharing)

| KEEP (canonical / thesis) | ARCHIVE or delete |
|---------------------------|-------------------|
| `Final Notebooks/` (ArSL letters) | Root-level duplicate `.h5` in `ArSL Letter (Arabic)/` |
| Canonical `.h5` per model | `Letters/Merger Notebook new approach/Trials/` |
| `ArSL_Word_Training_v2.ipynb` + `arsl_v2_*` | Old `ArSL_Word_Training.ipynb` |
| One ASL word training notebook | `ASL_Word_Training 1.ipynb`, all `*.backup*` |
| `Words/Shared/shared_word_vocabulary.csv` | `Testing repos/`, external base repos |
| `Separated Pipelines/` | One-off `scripts/*.py` patchers (`patch_*`, `fix_*`, `temp_*`) |

---

## 11. .gitignore Essentials (already partly set)

```gitignore
*.rar
*.mp4
__pycache__/
.ipynb_checkpoints/
node_modules/
# Large caches - keep local
Words/**/arsl_word_sequences*.npz
Words/**/*_word_sequences.npz
# Optional (uncomment for clean repo, share models externally):
# *.h5
# *.tflite
```

---

## 12. Quick Reference — Data Flow per Model

```
LETTERS:  image --MediaPipe--> 63 floats --> MLP --> letter
                 `------------> 128/96 crop --> MobileNetV2 --> letter
WORDS:    video --MediaPipe--> (30,63) --.npz--> BiLSTM --> word_id --> shared vocab --> EN/AR text
```
