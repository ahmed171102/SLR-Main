# 01 — Inventory: What We Have vs What We Need

> This document lists everything that already exists from the ML training phase
> and everything that needs to be created for deployment.

---

## WHAT WE ALREADY HAVE (Completed)

### Trained Models (.h5 files)

| # | Model | File | Shape In | Shape Out | Status |
|---|-------|------|----------|-----------|--------|
| 1 | ASL Letter MLP | `asl_mediapipe_mlp_model.h5` | (1, 63) | 29 classes (A-Z, space, del, nothing) | DONE |
| 2 | ArSL Letter MLP | `arsl_mediapipe_mlp_model_final.h5` | (1, 63) | Arabic letter classes | DONE |
| 3 | ASL Word BiLSTM | `asl_word_lstm_model_best.h5` | (30, 63) | 157 word classes | DONE |
| 4 | MobileNetV2 (optional) | `sign_language_model_MobileNetV2.h5` | (224,224,3) | 29 classes | DONE |

**Location of models (source paths):**
```
SLR Main/Letters/ASL Letter (English)/asl_mediapipe_mlp_model.h5
SLR Main/Letters/ArSL Letter (Arabic)/Final Notebooks/arsl_mediapipe_mlp_model_final.h5
SLR Main/Words/ASL Word (English)/asl_word_lstm_model_best.h5
SLR Main/Letters/ASL Letter (English)/sign_language_model_MobileNetV2.h5
```

### Label Encoder / Class Mapping Files

| # | File | Purpose | Location |
|---|------|---------|----------|
| 1 | `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (fit LabelEncoder) | `Letters/ASL Letter (English)/` |
| 2 | `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels (fit LabelEncoder) | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| 3 | `asl_word_classes.csv` | model_class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| 4 | `shared_word_vocabulary.csv` | word_id → english, arabic, category (157 rows) | `Words/Shared/` |

### Utility Code

| # | File | Purpose | Lines |
|---|------|---------|-------|
| 1 | `letter_stream_decoder.py` | Converts per-frame letter predictions to text with stabilization, cooldown, majority voting | 262 |

### Existing Documentation

| # | File | Location |
|---|------|----------|
| 1 | `DEPLOYMENT_GUIDE.md` | `Letters/Guides/` |
| 2 | `MODEL_SUMMARY.md` | `Words/Docs/` |
| 3 | `TEAM_QUICKSTART.md` | `Words/Docs/` |
| 4 | `ARCHITECTURE_AND_PIPELINE.md` | `Words/Docs/` |
| 5 | `LETTERS_WORDS_INTEGRATION.md` | `Words/Docs/` |
| 6 | `DATASET_GUIDE.md` | `Words/Docs/` |
| 7 | `WORD_BUILDING_GUIDE.md` | `Letters/Guides/` |
| 8 | `OPTIMIZATION_GUIDE.md` | `Letters/Guides/` |

### Training Notebooks (Reference)

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `Combined_Architecture.ipynb` | ASL letter live test (webcam) |
| 2 | `Mediapipe_Training.ipynb` | ASL letter MLP training |
| 3 | `ASL_Word_Training.ipynb` | ASL word BiLSTM training |
| 4 | `ASL_Word_Live_Test.ipynb` | ASL word live test (webcam) |
| 5 | `Combined_Architecture_Arabic_GPU.ipynb` | ArSL letter live test |
| 6 | `ArSL_Word_Training.ipynb` | ArSL word training (needs KArSL data) |

### Key Custom Code (Must Replicate in Backend)

**TemporalAttention Layer** (required to load word model):
```python
class TemporalAttention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight('att_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform')
        self.b = self.add_weight('att_bias', shape=(input_shape[1], 1), initializer='zeros')
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)
```

---

## WHAT WE NEED TO CREATE (New)

### Scripts (Phase 1)

| # | File | Purpose |
|---|------|---------|
| 1 | `scripts/copy_models.py` | Copy all .h5 + .csv files from SLR Main into `backend/model_files/` |
| 2 | `scripts/convert_models.py` | Convert .h5 → .tflite for mobile (with TemporalAttention custom op) |
| 3 | `scripts/export_labels.py` | Export label encoders to JSON for web/mobile use |

### Backend API (Phase 2) — ~12 files

| # | File | Purpose |
|---|------|---------|
| 1 | `backend/app/__init__.py` | Package init |
| 2 | `backend/app/main.py` | FastAPI app entry point |
| 3 | `backend/app/config.py` | Settings (model paths, thresholds, CORS) |
| 4 | `backend/app/models/loader.py` | Load all .h5 models + label encoders |
| 5 | `backend/app/models/letter_predictor.py` | Letter prediction (MLP, single frame) |
| 6 | `backend/app/models/word_predictor.py` | Word prediction (BiLSTM, 30-frame window) |
| 7 | `backend/app/models/mode_detector.py` | Motion-based letter/word mode switching |
| 8 | `backend/app/models/letter_decoder.py` | Letter stream → text (port of letter_stream_decoder.py) |
| 9 | `backend/app/models/word_decoder.py` | Word predictions → sentence builder |
| 10 | `backend/app/routes/predict.py` | REST endpoints (POST /predict/letter, POST /predict/word) |
| 11 | `backend/app/routes/websocket.py` | WebSocket endpoint for real-time streaming |
| 12 | `backend/requirements.txt` | Python dependencies |
| 13 | `backend/Dockerfile` | Container configuration |

### Web Frontend (Phase 3) — ~15 files

| # | File | Purpose |
|---|------|---------|
| 1 | `web/package.json` | Dependencies (React, MediaPipe, etc.) |
| 2 | `web/src/App.tsx` | Main app with routing |
| 3 | `web/src/pages/Home.tsx` | Landing page |
| 4 | `web/src/pages/Recognize.tsx` | Main recognition page |
| 5 | `web/src/components/Camera.tsx` | Webcam capture with MediaPipe |
| 6 | `web/src/components/HandOverlay.tsx` | Draw landmarks on video |
| 7 | `web/src/components/PredictionDisplay.tsx` | Show prediction results |
| 8 | `web/src/components/SentenceBuilder.tsx` | Display built sentence |
| 9 | `web/src/components/ModeIndicator.tsx` | Show letter/word mode |
| 10 | `web/src/components/LanguageToggle.tsx` | EN/AR switch |
| 11 | `web/src/hooks/useMediaPipe.ts` | MediaPipe hands hook |
| 12 | `web/src/hooks/useWebSocket.ts` | WebSocket connection hook |
| 13 | `web/src/services/api.ts` | API client |
| 14 | `web/src/utils/landmarks.ts` | Landmark processing utilities |
| 15 | `web/tailwind.config.js` | Tailwind CSS config |

### Mobile App (Phase 4) — ~12 files

| # | File | Purpose |
|---|------|---------|
| 1 | `mobile/package.json` | Expo + React Native deps |
| 2 | `mobile/App.tsx` | Entry point with navigation |
| 3 | `mobile/src/screens/HomeScreen.tsx` | Home screen |
| 4 | `mobile/src/screens/RecognizeScreen.tsx` | Camera + recognition |
| 5 | `mobile/src/screens/SettingsScreen.tsx` | Language, mode settings |
| 6 | `mobile/src/components/CameraView.tsx` | Camera with MediaPipe |
| 7 | `mobile/src/components/ResultOverlay.tsx` | Prediction overlay |
| 8 | `mobile/src/services/tfliteModel.ts` | TFLite model loader |
| 9 | `mobile/src/services/mediapipe.ts` | MediaPipe hand detection |
| 10 | `mobile/src/services/decoder.ts` | Letter/word decoders (JS port) |
| 11 | `mobile/src/utils/landmarks.ts` | Landmark processing |
| 12 | `mobile/assets/models/` | TFLite model files + label JSONs |

### Deployment Configs (Phase 5)

| # | File | Purpose |
|---|------|---------|
| 1 | `backend/Dockerfile` | Docker container for backend |
| 2 | `backend/railway.toml` | Railway deployment config |
| 3 | `web/vercel.json` | Vercel deployment config |
| 4 | `docker-compose.yml` | Local dev environment |
| 5 | `.env.example` | Environment variables template |

---

## Summary Count

| Category | Already Have | Need to Create |
|----------|-------------|---------------|
| ML Models | 4 files | 0 (convert to TFLite: 3 files) |
| Data/Labels | 4 files | 3 (JSON exports) |
| Utility Code | 1 file | 0 (port to backend) |
| Documentation | 8 files | 11 guide files (these docs) |
| Scripts | 0 | 3 |
| Backend API | 0 | ~13 files |
| Web Frontend | 0 | ~15 files |
| Mobile App | 0 | ~12 files |
| Deploy Config | 0 | ~5 files |
| **TOTAL** | **17 files** | **~62 files** |

> The models are the hard part — and those are DONE.
> The 62 new files are mostly boilerplate code connecting your models to a UI.
