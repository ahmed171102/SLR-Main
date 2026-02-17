# ESHARA — Full Deployment Plan (Master Document)

> **Project**: ESHARA Sign Language Recognition System
> **Date**: February 2026
> **Platforms**: Web + Mobile
> **Languages**: Bilingual (English ASL + Arabic ArSL)
> **Modes**: Letters + Words

---

## Table of Contents

| # | Guide File | Topic |
|---|-----------|-------|
| 1 | [01_INVENTORY.md](01_INVENTORY.md) | What We Have vs What We Need to Create |
| 2 | [02_TECH_STACK.md](02_TECH_STACK.md) | Languages, Frameworks & Tools |
| 3 | [03_ACCOUNTS_SETUP.md](03_ACCOUNTS_SETUP.md) | Accounts, Databases & Services Setup |
| 4 | [04_FOLDER_STRUCTURE.md](04_FOLDER_STRUCTURE.md) | Complete Project Folder Structure |
| 5 | [05_BACKEND_GUIDE.md](05_BACKEND_GUIDE.md) | Backend API — Step-by-Step Build Guide |
| 6 | [06_WEB_FRONTEND_GUIDE.md](06_WEB_FRONTEND_GUIDE.md) | Web Frontend — Step-by-Step Build Guide |
| 7 | [07_MOBILE_APP_GUIDE.md](07_MOBILE_APP_GUIDE.md) | Mobile App — Step-by-Step Build Guide |
| 8 | [08_MODEL_CONVERSION.md](08_MODEL_CONVERSION.md) | Model Conversion (.h5 → TFLite) |
| 9 | [09_DEPLOYMENT_CLOUD.md](09_DEPLOYMENT_CLOUD.md) | Cloud Deployment & Docker |
| 10 | [10_TESTING_CHECKLIST.md](10_TESTING_CHECKLIST.md) | Testing & Verification Checklist |
| 11 | [11_TIMELINE.md](11_TIMELINE.md) | Timeline, Difficulty & Comparison |

---

## Quick Summary

### What Exists (Done)
- 4 trained ML models (.h5 files) for ASL letters, ArSL letters, ASL words
- Training notebooks, datasets, label encoders
- Existing docs (MODEL_SUMMARY, DEPLOYMENT_GUIDE, TEAM_QUICKSTART, etc.)
- `letter_stream_decoder.py` utility
- `shared_word_vocabulary.csv` (157 bilingual words)
- `asl_word_classes.csv` (word class mappings)

### What We Build (New)
- Python backend API (FastAPI + WebSocket)
- Web frontend (React + TypeScript + MediaPipe JS)
- Mobile app (React Native + Expo + TFLite)
- TFLite model conversions
- Docker configuration
- Cloud deployment (Railway + Vercel)
- Optional: Supabase auth + database

### Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   WEBCAM /   │     │   MEDIAPIPE  │     │   ML MODEL   │
│   CAMERA     │────▶│   HANDS      │────▶│   PREDICT    │
│              │     │  (landmarks) │     │  (letter/word)│
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │   DECODER    │
                                          │  (text build) │
                                          └──────────────┘

WEB:    Camera → MediaPipe JS (browser) → WebSocket → FastAPI → Model → Response
MOBILE: Camera → MediaPipe (on-device) → TFLite (on-device) → Result (100% offline)
```

### Build Order

```
Phase 0: Setup          → Accounts, folders, dependencies
Phase 1: Scripts        → copy_models.py, convert_models.py
Phase 2: Backend API    → FastAPI + WebSocket + all endpoints
Phase 3: Web Frontend   → React app with camera + MediaPipe
Phase 4: Mobile App     → React Native + Expo + TFLite
Phase 5: Deploy         → Docker, Railway, Vercel
Phase 6: Polish         → Auth, history, settings, Arabic UI
```

### Difficulty Comparison

| Task | Difficulty | Time |
|------|-----------|------|
| Training ML models (what you already did) | ⭐⭐⭐⭐⭐ Hard | Weeks |
| Backend API | ⭐⭐⭐ Medium | 3-4 days |
| Web Frontend | ⭐⭐⭐ Medium | 4-5 days |
| Mobile App | ⭐⭐⭐⭐ Medium-Hard | 5-7 days |
| Cloud Deployment | ⭐⭐ Easy | 1-2 days |
| **Total Deployment** | **⭐⭐⭐ Medium** | **~16-23 days** |

> **Bottom line**: The hardest part (model training) is DONE. Deployment is mostly connecting pieces together.

---

## Model Files Reference

| Model | File | Location | Input | Output |
|-------|------|----------|-------|--------|
| ASL Letters MLP | `asl_mediapipe_mlp_model.h5` | `Letters/ASL Letter (English)/` | (1, 63) | 29 classes |
| ArSL Letters MLP | `arsl_mediapipe_mlp_model_final.h5` | `Letters/ArSL Letter (Arabic)/Final Notebooks/` | (1, 63) | Arabic letter classes |
| ASL Words BiLSTM | `asl_word_lstm_model_best.h5` | `Words/ASL Word (English)/` | (30, 63) | 157 word classes |
| MobileNetV2 (optional) | `sign_language_model_MobileNetV2.h5` | `Letters/ASL Letter (English)/` | (224,224,3) | 29 classes |

## Label/Data Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter labels (LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter labels (LabelEncoder) | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class → word_id mapping | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | Word_id → English + Arabic names | `Words/Shared/` |
| `letter_stream_decoder.py` | Letter prediction → text utility | `Letters/Guides/` |

---

*Read each numbered guide file for detailed step-by-step instructions.*
