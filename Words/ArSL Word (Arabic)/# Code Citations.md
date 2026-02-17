# Code Citations

## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```


## License: unknown
https://github.com/onyd/Iris/blob/6ba3cae6ebe1c95d655dac24d136e9d87d4aa406/DL/Models/AttentionLayer.py

```
Now I have everything. Here's the full deployment plan markdown file:

```markdown
# Sign Language Recognition — Full Deployment Plan

> **Project:** Bilingual Sign Language Recognition (ASL + ArSL)  
> **Platforms:** Web Application + Mobile Application  
> **Date:** February 2026  
> **Status:** Models trained ✅ → Deployment NOT started ❌

---

## Table of Contents

1. [What We Already Have](#1-what-we-already-have)
2. [What We Need to Create](#2-what-we-need-to-create)
3. [Is Deployment Harder Than Models?](#3-is-deployment-harder-than-models)
4. [Languages & Technologies Needed](#4-languages--technologies-needed)
5. [Accounts & Services to Create](#5-accounts--services-to-create)
6. [Database Design](#6-database-design)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Folder Structure](#8-folder-structure)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Verification Checklist](#10-verification-checklist)
11. [Timeline Estimate](#11-timeline-estimate)

---

## 1. What We Already Have

### ✅ Trained Models (Ready to Deploy)

| Model | File | Input | Output | Location |
|---|---|---|---|---|
| ASL Letter (English) | `asl_mediapipe_mlp_model.h5` | `(1, 63)` single frame | 29 classes (A-Z + space/del/nothing) | `Letters/ASL Letter (English)/` |
| ArSL Letter (Arabic) | `arsl_mediapipe_mlp_model_final.h5` | `(1, 63)` single frame | 28+ Arabic letter classes | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| ASL Word (English) | `asl_word_lstm_model_best.h5` | `(30, 63)` video sequence | 157 word classes | `Words/ASL Word (English)/` |

### ✅ Supporting Data Files

| File | Purpose | Location |
|---|---|---|
| `asl_mediapipe_keypoints_dataset.csv` | ASL letter class labels (for LabelEncoder) | `Letters/ASL Letter (English)/` |
| `FINAL_CLEAN_DATASET.csv` | ArSL letter class labels | `Letters/ArSL Letter (Arabic)/Final Notebooks/` |
| `asl_word_classes.csv` | Word model class_index → word_id (158 rows) | `Words/ASL Word (English)/` |
| `shared_word_vocabulary.csv` | 157 bilingual words: word_id → english + arabic + category | `Words/Shared/` |

### ✅ Existing Code (Reusable)

| Component | File | Lines | What It Does |
|---|---|---|---|
| Letter Stream Decoder | `letter_stream_decoder.py` | 262 | Converts per-frame predictions into text (stability window, cooldown, space/del handling) |
| TemporalAttention Layer | Defined in `ASL_Word_Training.ipynb` | ~15 | Custom Keras layer needed to load the word model |
| Live webcam letter test | `Combined_Architecture.ipynb` | 840 | Letter recognition with webcam (MLP + MediaPipe) |
| Live webcam word test | `ASL_Word_Live_Test.ipynb` | 481 | Word recognition with webcam (BiLSTM + sliding window) |
| Mode switching design | `LETTERS_WORDS_INTEGRATION.md` | 232 | Architecture doc for combining letters + words |
| Deployment concepts | `DEPLOYMENT_GUIDE.md` | 394 | Overview of deployment options (no actual code) |

### ✅ Documentation

- `ARCHITECTURE_AND_PIPELINE.md` — Full data flow diagram
- `MODEL_SUMMARY.md` — Model specs and hyperparameters
- `TEAM_QUICKSTART.md` — How to run training notebooks
- `DATASET_GUIDE.md` — Dataset details
- Multiple optimization guides in `Letters/Guides/`

### ❌ What We Do NOT Have Yet

- No backend API (no Flask, FastAPI, or any server)
- No frontend (no React, no web UI)
- No mobile app
- No database
- No user authentication
- No Docker configuration
- No TFLite converted models
- No TypeScript/JavaScript code at all
- No deployment to any cloud
- No CI/CD pipeline

---

## 2. What We Need to Create

### Summary: 3 Major Systems to Build

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM 1: BACKEND API                                       │
│  Language: Python                                            │
│  Framework: FastAPI                                          │
│  What: REST + WebSocket server that runs the models          │
│  Files to create: ~15 Python files                           │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 2: WEB FRONTEND                                      │
│  Language: TypeScript + React                                │
│  Framework: Vite + Tailwind CSS                              │
│  What: Browser app with webcam + live predictions            │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★☆☆ (Medium)                                  │
├─────────────────────────────────────────────────────────────┤
│  SYSTEM 3: MOBILE APP                                        │
│  Language: TypeScript + React Native                         │
│  Framework: Expo                                             │
│  What: Android/iOS app with on-device offline inference      │
│  Files to create: ~15 TypeScript files                       │
│  Difficulty: ★★★★☆ (Hard — TFLite integration is tricky)     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed File-by-File Creation List

#### Backend (Python — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/main.py` | FastAPI app entry, CORS, startup | Easy | 30 min |
| 2 | `app/config.py` | All settings, paths, thresholds | Easy | 20 min |
| 3 | `app/schemas.py` | Pydantic request/response models | Easy | 30 min |
| 4 | `app/models/loader.py` | Load all .h5 models + encoders at startup | Medium | 1 hr |
| 5 | `app/models/letter_predictor.py` | Single-frame MLP inference | Easy | 30 min |
| 6 | `app/models/word_predictor.py` | 30-frame BiLSTM inference | Medium | 45 min |
| 7 | `app/models/mode_detector.py` | Motion analysis: still→letter, moving→word | Medium | 1 hr |
| 8 | `app/core/letter_decoder.py` | Copy existing LetterStreamDecoder | Easy | 15 min |
| 9 | `app/core/word_decoder.py` | Word stability + cooldown logic | Medium | 45 min |
| 10 | `app/core/sentence_builder.py` | Combine letter + word outputs | Medium | 1 hr |
| 11 | `app/core/session_manager.py` | Per-WebSocket session state | Medium | 45 min |
| 12 | `app/routes/predict.py` | POST /api/predict/letter endpoint | Easy | 30 min |
| 13 | `app/routes/predict_word.py` | POST /api/predict/word endpoint | Easy | 30 min |
| 14 | `app/routes/ws_combined.py` | WebSocket /api/ws/combined (real-time) | Hard | 2 hr |
| 15 | `app/routes/health.py` | GET /health endpoint | Easy | 10 min |
| 16 | `requirements.txt` | Python dependencies | Easy | 5 min |
| 17 | `Dockerfile` | Container configuration | Medium | 30 min |

#### Web Frontend (TypeScript/React — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `src/App.tsx` | Main layout + routing | Easy | 20 min |
| 2 | `src/pages/Home.tsx` | Camera + predictions + sentence page | Medium | 1 hr |
| 3 | `src/hooks/useMediaPipe.ts` | MediaPipe Hands JS setup + landmark extraction | Hard | 2 hr |
| 4 | `src/hooks/useWebSocket.ts` | WS connection to backend | Medium | 1 hr |
| 5 | `src/hooks/useSentence.ts` | Sentence state management | Easy | 30 min |
| 6 | `src/components/CameraFeed.tsx` | Webcam + canvas overlay | Hard | 2 hr |
| 7 | `src/components/PredictionDisplay.tsx` | Current letter/word + confidence | Easy | 45 min |
| 8 | `src/components/ModeIndicator.tsx` | LETTER / WORD / IDLE mode badge | Easy | 20 min |
| 9 | `src/components/SentenceBar.tsx` | Built sentence (English + Arabic) | Medium | 45 min |
| 10 | `src/components/LanguageToggle.tsx` | ASL ↔ ArSL switch | Easy | 20 min |
| 11 | `src/components/ConfidenceBar.tsx` | Visual confidence meter | Easy | 20 min |
| 12 | `src/components/StabilityMeter.tsx` | Hold progress / buffer fill | Easy | 20 min |
| 13 | `src/components/TopPredictions.tsx` | Top-3 predictions list | Easy | 20 min |
| 14 | `src/services/api.ts` | REST + WS client config | Easy | 20 min |
| 15 | `src/utils/landmarks.ts` | Flatten 21 landmarks → 63 floats | Easy | 15 min |

#### Mobile App (TypeScript/React Native — ~15 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `app/(tabs)/index.tsx` | Main camera recognition screen | Hard | 3 hr |
| 2 | `app/(tabs)/settings.tsx` | Language, thresholds, camera | Medium | 1 hr |
| 3 | `app/(tabs)/history.tsx` | Saved sentences | Easy | 45 min |
| 4 | `app/_layout.tsx` | Tab navigation layout | Easy | 20 min |
| 5 | `components/CameraView.tsx` | Expo Camera + frame processing | Hard | 3 hr |
| 6 | `components/HandOverlay.tsx` | Draw landmarks on camera | Medium | 1 hr |
| 7 | `components/PredictionBanner.tsx` | Current letter/word + confidence | Easy | 30 min |
| 8 | `components/ModeChip.tsx` | Mode indicator | Easy | 15 min |
| 9 | `components/SentenceDisplay.tsx` | Bilingual sentence bar | Medium | 45 min |
| 10 | `services/mediapipeHands.ts` | On-device MediaPipe hand detection | Hard | 2 hr |
| 11 | `services/tfliteInference.ts` | Run TFLite models on-device | Hard | 3 hr |
| 12 | `services/modeDetector.ts` | Motion-based letter↔word switching | Medium | 1 hr |
| 13 | `services/letterDecoder.ts` | TS port of LetterStreamDecoder | Medium | 1.5 hr |
| 14 | `services/wordDecoder.ts` | TS port of word stability logic | Medium | 1 hr |
| 15 | `services/sentenceBuilder.ts` | Combine letter + word outputs | Medium | 45 min |

#### Scripts & Docs (~7 files)

| # | File | Purpose | Difficulty | Est. Time |
|---|---|---|---|---|
| 1 | `scripts/copy_models.py` | Copy .h5/.csv from training folders | Easy | 15 min |
| 2 | `scripts/convert_all_tflite.py` | Convert 3 models to .tflite | Medium | 1 hr |
| 3 | `scripts/test_api.py` | Automated API testing | Easy | 30 min |
| 4 | `docs/DEPLOYMENT_README.md` | Master setup guide | Easy | 1 hr |
| 5 | `docs/API_REFERENCE.md` | Endpoint documentation | Easy | 45 min |
| 6 | `docs/ARCHITECTURE.md` | System architecture doc | Easy | 30 min |
| 7 | `docs/SETUP_GUIDE.md` | Step-by-step per platform | Easy | 1 hr |

---

## 3. Is Deployment Harder Than Models?

### Honest Comparison

| Aspect | Model Training | Deployment |
|---|---|---|
| **Difficulty** | ★★★★☆ | ★★★☆☆ |
| **Complexity** | Deep math, architecture design, hyperparameter tuning | Connecting systems, API design, UI components |
| **Time** | Weeks-months (data collection + training) | 2-4 weeks (building + testing) |
| **Skills needed** | Python, ML/DL, MediaPipe, TensorFlow | Python, TypeScript, React, React Native, Docker |
| **Hardest part** | Getting good accuracy | Making real-time webcam smooth + TFLite conversion |
| **Risk of failure** | High (model might not learn) | Low (standard web/mobile patterns) |
| **Debugging** | Hard (why is accuracy low?) | Easier (error messages are clear) |
| **New skills to learn** | You already know this | FastAPI, React, React Native, Docker (possibly new) |

### Verdict

**Model training was harder intellectually** (ML is complex). **Deployment is harder practically** because:
- You need to learn **3 new frameworks** (FastAPI, React, React Native)
- You need to manage **accounts, servers, databases** (infrastructure)
- You need to make **real-time webcam work smoothly** in a browser and phone
- TFLite conversion of the BiLSTM with custom TemporalAttention layer is tricky

**But deployment is more predictable** — there's a clear path from A to B. Models can fail in mysterious ways; deployment either works or gives you a clear error.

---

## 4. Languages & Technologies Needed

### Languages You'll Write Code In

| Language | Where Used | Amount | Need to Learn? |
|---|---|---|---|
| **Python 3.9** | Backend API, scripts, model conversion | ~40% of code | Already know ✅ |
| **TypeScript** | Web frontend, mobile app | ~55% of code | Need to learn ⚠️ |
| **HTML/CSS** | Web frontend (via React JSX + Tailwind) | ~5% of code | Basic knowledge enough |
| **SQL** | Database queries (if adding auth) | Very little | Basic only |

### Frameworks & Libraries

| Technology | What It Is | What It Does For Us |
|---|---|---|
| **FastAPI** (Python) | Modern web API framework | Serves our models as REST + WebSocket endpoints |
| **Uvicorn** (Python) | ASGI server | Runs FastAPI with async support |
| **TensorFlow 2.10** (Python) | ML framework | Loads and runs our .h5 models |
| **React 18** (TypeScript) | UI library | Builds the web frontend |
| **Vite** | Build tool | Fast React development server |
| **Tailwind CSS** | CSS framework | Styles the web UI without writing CSS |
| **MediaPipe JS** | Hand detection (browser) | Runs hand detection client-side in the browser |
| **React Native** (TypeScript) | Mobile framework | Builds Android + iOS apps from one codebase |
| **Expo** | React Native tooling | Simplifies building, testing, deploying mobile apps |
| **TFLite** | Mobile ML runtime | Runs our models on-device (phone) |
| **Docker** | Containerization | Packages backend for cloud deployment |
| **PostgreSQL** (optional) | Database | Stores users, sessions, sentence history |

### What Runs Where

```
BROWSER (Client-Side):
  - React (TypeScript) — UI components
  - MediaPipe Hands JS — hand detection (NO video sent to server)
  - WebSocket client — sends 63 float landmarks per frame

SERVER (Backend):
  - FastAPI (Python) — API endpoints
  - TensorFlow — loads .h5 models, runs inference
  - LetterStreamDecoder — per-session sentence building
  - PostgreSQL — user data (optional)

PHONE (On-Device):
  - React Native (TypeScript) — UI
  - MediaPipe Hands (mobile SDK) — hand detection
  - TFLite — runs .tflite models locally
  - Everything offline — no server needed
```

---

## 5. Accounts & Services to Create

### Required Accounts (FREE)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 1 | **GitHub** | github.com | Code hosting, version control | Free |
| 2 | **Node.js** | nodejs.org | Install npm for React/React Native | Free (download) |
| 3 | **Expo** | expo.dev | Build mobile app APK/IPA without Android Studio | Free tier |

### Required Accounts for Deployment (FREE tiers available)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 4 | **Railway** | railway.app | Host the FastAPI backend | Free $5/month credit |
| 5 | **Vercel** | vercel.com | Host the React web frontend | Free tier |
| 6 | **Supabase** | supabase.com | PostgreSQL database + auth (optional) | Free tier (500MB) |

### Alternative Hosting Options

| Service | Backend | Frontend | Database | Free Tier |
|---|---|---|---|---|
| **Railway** | ✅ Docker | ❌ | ✅ PostgreSQL | $5/mo credit |
| **Render** | ✅ Docker | ✅ Static | ✅ PostgreSQL | 750 hrs/mo |
| **Vercel** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Netlify** | ❌ (no Python) | ✅ React | ❌ | Unlimited |
| **Supabase** | ❌ | ❌ | ✅ PostgreSQL + Auth | 500MB |
| **AWS EC2** | ✅ anything | ✅ S3 | ✅ RDS | 12 months free |
| **Google Cloud Run** | ✅ Docker | ✅ Firebase | ✅ Cloud SQL | $300 credit |

### Recommended Stack (Cheapest)

```
Backend API  → Railway (free $5 credit, auto-deploy from GitHub)
Web Frontend → Vercel (free, auto-deploy from GitHub)
Database     → Supabase (free PostgreSQL + built-in auth)
Mobile Build → Expo EAS (free for dev builds)
```

### Accounts for Mobile App Publishing (Optional — costs money)

| # | Service | URL | What For | Cost |
|---|---|---|---|---|
| 7 | **Google Play Console** | play.google.com/console | Publish Android app | $25 one-time |
| 8 | **Apple Developer** | developer.apple.com | Publish iOS app | $99/year |

### Software to Install on Your Computer

| # | Software | Version | Install Command / URL |
|---|---|---|---|
| 1 | Python | 3.9.x | Already installed ✅ |
| 2 | Node.js | 18+ (LTS) | https://nodejs.org → download LTS |
| 3 | npm | comes with Node.js | Automatic with Node.js |
| 4 | Git | latest | https://git-scm.com |
| 5 | VS Code | latest | Already using ✅ |
| 6 | Docker Desktop | latest | https://docker.com (optional, for deployment) |
| 7 | Expo Go app | latest | Install on your phone from App Store/Play Store |

---

## 6. Database Design

### Do You NEED a Database?

| Feature | Without Database | With Database |
|---|---|---|
| Real-time sign prediction | ✅ Works | ✅ Works |
| Sentence building | ✅ Works (in memory) | ✅ Works |
| Bilingual display | ✅ Works (from CSV) | ✅ Works |
| User accounts / login | ❌ No | ✅ Yes |
| Save sentence history | ❌ Lost on refresh | ✅ Persistent |
| Usage analytics | ❌ No | ✅ Yes |
| Multiple users | ❌ No sessions | ✅ Yes |

**Recommendation:** Start WITHOUT a database. Add Supabase later if you need users/history.

### Database Schema (If Using Supabase/PostgreSQL)

```sql
-- USERS TABLE
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'asl',  -- 'asl' or 'arsl'
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

-- SESSIONS TABLE (each time user opens the app/web)
CREATE TABLE sessions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    started_at    TIMESTAMP DEFAULT NOW(),
    ended_at      TIMESTAMP,
    language_used VARCHAR(10),
    platform      VARCHAR(20)  -- 'web', 'android', 'ios'
);

-- SENTENCES TABLE (saved recognized sentences)
CREATE TABLE sentences (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
    text_english  TEXT NOT NULL,
    text_arabic   TEXT,
    word_count    INTEGER DEFAULT 0,
    letter_count  INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- PREDICTIONS LOG (optional — analytics)
CREATE TABLE prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(id) ON DELETE SET NULL,
    mode          VARCHAR(10),  -- 'letter' or 'word'
    prediction    VARCHAR(100),
    confidence    FLOAT,
    language      VARCHAR(10),
    timestamp     TIMESTAMP DEFAULT NOW()
);
```

### Setting Up Supabase (If You Want Auth + Database)

1. Go to https://supabase.com → Sign up with GitHub
2. Click "New Project" → name it `slr-app` → choose region → set database password
3. Go to SQL Editor → paste the schema above → click "Run"
4. Go to Authentication → Enable email/password sign-up
5. Go to Settings → API → copy:
   - `SUPABASE_URL` (e.g., `https://abc123.supabase.co`)
   - `SUPABASE_ANON_KEY` (public key for frontend)
   - `SUPABASE_SERVICE_KEY` (secret key for backend)
6. Install in your projects:
   - Backend: `pip install supabase`
   - Web: `npm install @supabase/supabase-js`
   - Mobile: `npm install @supabase/supabase-js`

---

## 7. Step-by-Step Build Guide

### PHASE 0: Setup & Prerequisites (Day 1)

#### Step 0.1 — Install Node.js
```bash
# Download from https://nodejs.org (LTS version, 18+)
# After install, verify:
node --version    # Should show v18.x or v20.x
npm --version     # Should show 9.x or 10.x
```

#### Step 0.2 — Install Git (if not already)
```bash
# Download from https://git-scm.com
git --version     # Should show 2.x
```

#### Step 0.3 — Create GitHub Repository
1. Go to https://github.com → Sign in → "New Repository"
2. Name: `sign-language-app`
3. Private (your graduation project)
4. Clone it:
```bash
cd "m:\Term 10\Grad"
git clone https://github.com/YOUR_USERNAME/sign-language-app.git Deployment
cd Deployment
```

#### Step 0.4 — Create the folder structure
```bash
mkdir backend backend\app backend\app\models backend\app\routes backend\app\core backend\app\utils backend\model_files backend\scripts
mkdir web
mkdir mobile
mkdir scripts
mkdir docs
```

#### Step 0.5 — Copy model files
Copy these files into `Deployment\backend\model_files\`:
- From `SLR Main\Letters\ASL Letter (English)\`:
  - `asl_mediapipe_mlp_model.h5`
  - `asl_mediapipe_keypoints_dataset.csv`
- From `SLR Main\Letters\ArSL Letter (Arabic)\Final Notebooks\`:
  - `arsl_mediapipe_mlp_model_final.h5`
  - `FINAL_CLEAN_DATASET.csv`
- From `SLR Main\Words\ASL Word (English)\`:
  - `asl_word_lstm_model_best.h5`
  - `asl_word_classes.csv`
- From `SLR Main\Words\Shared\`:
  - `shared_word_vocabulary.csv`

---

### PHASE 1: Backend API (Days 2–5)

#### Step 1.1 — Set up Python environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install fastapi uvicorn[standard] tensorflow==2.10.0 numpy pandas scikit-learn websockets python-multipart arabic-reshaper python-bidi
pip freeze > requirements.txt
```

#### Step 1.2 — Create `app/__init__.py` (empty file)
```python
# empty — makes this a Python package
```

#### Step 1.3 — Create `app/config.py`
Define all settings:
- `MODEL_DIR` pointing to `model_files/`
- Letter model filenames for ASL + ArSL
- Word model filename
- CSV filenames for encoders and vocabulary
- Thresholds: `LETTER_MIN_CONFIDENCE=0.7`, `LETTER_STABLE_WINDOW=5`, `LETTER_COOLDOWN=0.6`
- Word: `SEQUENCE_LENGTH=30`, `NUM_FEATURES=63`, `WORD_CONFIDENCE=0.35`, `WORD_STABILITY=3`, `WORD_COOLDOWN=2.0`
- Mode detection: `MOTION_THRESHOLD=0.015`, `MODE_SWITCH_DELAY=0.5`

#### Step 1.4 — Create `app/models/loader.py`
At startup:
1. Load ASL letter MLP: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")`
2. Load ArSL letter MLP: `tf.keras.models.load_model("arsl_mediapipe_mlp_model_final.h5")`
3. Define `TemporalAttention` class (copy from training notebook — 15 lines)
4. Load ASL word BiLSTM: `tf.keras.models.load_model("asl_word_lstm_model_best.h5", custom_objects={"TemporalAttention": TemporalAttention})`
5. Fit `LabelEncoder` from ASL CSV (column "label")
6. Fit `LabelEncoder` from ArSL CSV (column "label")
7. Load `asl_word_classes.csv` → dict of `model_class_index → word_id`
8. Load `shared_word_vocabulary.csv` → dict of `word_id → {english, arabic, category}`

**Key gotcha:** The `TemporalAttention` layer MUST be defined identically to training:
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                    shape=(input_shape[-1], 1),
                    initializer='glorot_uniform',
```

