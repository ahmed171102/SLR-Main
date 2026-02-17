# 04 — Complete Project Folder Structure

> The exact folder layout to create in `m:\Term 10\Grad\Deployment\`.

---

## Full Structure

```
Deployment/
│
├── docs/                              # Documentation (these guide files)
│   ├── FULL_DEPLOYMENT_PLAN.md
│   ├── 01_INVENTORY.md
│   ├── 02_TECH_STACK.md
│   ├── 03_ACCOUNTS_SETUP.md
│   ├── 04_FOLDER_STRUCTURE.md         ← You are here
│   ├── 05_BACKEND_GUIDE.md
│   ├── 06_WEB_FRONTEND_GUIDE.md
│   ├── 07_MOBILE_APP_GUIDE.md
│   ├── 08_MODEL_CONVERSION.md
│   ├── 09_DEPLOYMENT_CLOUD.md
│   ├── 10_TESTING_CHECKLIST.md
│   └── 11_TIMELINE.md
│
├── scripts/                           # Setup & conversion scripts
│   ├── copy_models.py                 # Copy .h5 + .csv from SLR Main
│   ├── convert_models.py             # .h5 → .tflite conversion
│   └── export_labels.py             # Export label encoders to JSON
│
├── backend/                           # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── config.py                 # Settings & environment vars
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py            # Load .h5 models + label encoders
│   │   │   ├── temporal_attention.py # Custom Keras layer (for word model)
│   │   │   ├── letter_predictor.py  # Single-frame letter prediction
│   │   │   ├── word_predictor.py    # 30-frame word prediction
│   │   │   ├── mode_detector.py     # Motion-based letter/word switch
│   │   │   ├── letter_decoder.py    # Letter → text stream builder
│   │   │   └── word_decoder.py      # Word → sentence builder
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py           # REST: POST /predict/letter, /predict/word
│   │   │   ├── websocket.py         # WS: /ws/recognize (real-time)
│   │   │   └── health.py            # GET /health
│   │   └── schemas/
│   │       ├── __init__.py
│   │       └── prediction.py        # Pydantic request/response models
│   ├── model_files/                  # Copied model & label files
│   │   ├── asl_mediapipe_mlp_model.h5
│   │   ├── arsl_mediapipe_mlp_model_final.h5
│   │   ├── asl_word_lstm_model_best.h5
│   │   ├── sign_language_model_MobileNetV2.h5  (optional)
│   │   ├── asl_mediapipe_keypoints_dataset.csv
│   │   ├── FINAL_CLEAN_DATASET.csv
│   │   ├── asl_word_classes.csv
│   │   ├── shared_word_vocabulary.csv
│   │   ├── asl_letter_labels.json    # Exported label encoder
│   │   ├── arsl_letter_labels.json   # Exported label encoder
│   │   └── word_labels.json          # Exported word mappings
│   ├── tests/
│   │   ├── test_letter_predictor.py
│   │   ├── test_word_predictor.py
│   │   └── test_websocket.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   └── railway.toml
│
├── web/                               # React web frontend
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── App.tsx                   # Root component + routing
│   │   ├── main.tsx                  # Entry point
│   │   ├── index.css                 # Global styles + Tailwind
│   │   ├── pages/
│   │   │   ├── Home.tsx              # Landing page
│   │   │   ├── Recognize.tsx         # Main recognition page
│   │   │   ├── History.tsx           # Translation history (optional)
│   │   │   └── About.tsx             # About page
│   │   ├── components/
│   │   │   ├── Camera.tsx            # Webcam + MediaPipe hands
│   │   │   ├── HandOverlay.tsx       # Draw hand landmarks on canvas
│   │   │   ├── PredictionDisplay.tsx # Show current prediction
│   │   │   ├── SentenceBuilder.tsx   # Display built text/sentence
│   │   │   ├── ModeIndicator.tsx     # Letter/Word mode badge
│   │   │   ├── LanguageToggle.tsx    # EN ↔ AR switch
│   │   │   ├── ConfidenceBar.tsx     # Confidence visualization
│   │   │   └── Navbar.tsx            # Navigation bar
│   │   ├── hooks/
│   │   │   ├── useMediaPipe.ts       # Initialize & run MediaPipe
│   │   │   ├── useWebSocket.ts       # WebSocket connection + messages
│   │   │   └── useLanguage.ts        # i18n language hook
│   │   ├── services/
│   │   │   ├── api.ts                # Axios/fetch API client
│   │   │   └── supabase.ts           # Supabase client (optional)
│   │   ├── utils/
│   │   │   ├── landmarks.ts          # Normalize 21 landmarks → 63 floats
│   │   │   └── constants.ts          # API URLs, thresholds, config
│   │   ├── i18n/
│   │   │   ├── en.json               # English translations
│   │   │   └── ar.json               # Arabic translations
│   │   └── types/
│   │       └── index.ts              # TypeScript type definitions
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── vite.config.ts
│   └── vercel.json
│
├── mobile/                            # React Native + Expo mobile app
│   ├── App.tsx                       # Entry with navigation
│   ├── src/
│   │   ├── screens/
│   │   │   ├── HomeScreen.tsx
│   │   │   ├── RecognizeScreen.tsx
│   │   │   ├── HistoryScreen.tsx
│   │   │   └── SettingsScreen.tsx
│   │   ├── components/
│   │   │   ├── CameraView.tsx        # Camera + hand detection
│   │   │   ├── ResultOverlay.tsx     # Prediction display overlay
│   │   │   ├── SentenceDisplay.tsx   # Built sentence
│   │   │   └── ModeSwitch.tsx        # Letter/Word toggle
│   │   ├── services/
│   │   │   ├── tfliteModel.ts        # Load & run TFLite models
│   │   │   ├── mediapipe.ts          # Hand landmark detection
│   │   │   ├── letterDecoder.ts      # Letter stream decoder (JS port)
│   │   │   └── wordDecoder.ts        # Word sentence builder (JS port)
│   │   ├── utils/
│   │   │   ├── landmarks.ts          # Landmark normalization
│   │   │   └── constants.ts          # Config values
│   │   └── i18n/
│   │       ├── en.json
│   │       └── ar.json
│   ├── assets/
│   │   └── models/
│   │       ├── asl_letter_model.tflite
│   │       ├── arsl_letter_model.tflite
│   │       ├── asl_word_model.tflite
│   │       ├── asl_letter_labels.json
│   │       ├── arsl_letter_labels.json
│   │       └── word_labels.json
│   ├── package.json
│   ├── tsconfig.json
│   ├── app.json                      # Expo config
│   └── eas.json                      # EAS Build config
│
├── docker-compose.yml                 # Local dev: backend + optional DB
├── .gitignore
├── .env.example
└── README.md                          # Project overview
```

---

## Folder Creation Commands

Run these in terminal to create the folder skeleton:

```powershell
# Navigate to deployment folder
cd "m:\Term 10\Grad\Deployment"

# Create all directories
$dirs = @(
    "docs",
    "scripts",
    "backend/app/models",
    "backend/app/routes",
    "backend/app/schemas",
    "backend/model_files",
    "backend/tests",
    "web/public",
    "web/src/pages",
    "web/src/components",
    "web/src/hooks",
    "web/src/services",
    "web/src/utils",
    "web/src/i18n",
    "web/src/types",
    "mobile/src/screens",
    "mobile/src/components",
    "mobile/src/services",
    "mobile/src/utils",
    "mobile/src/i18n",
    "mobile/assets/models"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "Created: $dir"
}

Write-Host "`nAll directories created!"
```

---

## .gitignore File

```gitignore
# Environment
.env
.env.local
.env.production

# Python
__pycache__/
*.pyc
*.pyo
venv/
.venv/

# Node
node_modules/
dist/
build/
.next/

# Model files (too large for git)
*.h5
*.tflite
*.npz
backend/model_files/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Expo
.expo/
*.jks
*.p8
*.p12
*.key
*.mobileprovision
*.orig.*
```

> **Important**: Model files (.h5, .tflite) are too large for GitHub. They stay local and get copied via `copy_models.py`. For production, they're baked into the Docker image.
