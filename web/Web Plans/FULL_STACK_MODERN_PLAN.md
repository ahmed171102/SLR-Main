# Full-Stack Modern Plan (Graduation Project) — SLR (ASL/ArSL)

**Date:** 2026-04-13  
**Style:** Modern Dark UI (React + Tailwind)  
**Architecture:** MediaPipe in-browser → WebSocket → FastAPI → Model inference → UI updates

---

## 1) What you are building (final deliverable)

A full-stack web application that:
- Opens the webcam in the browser
- Runs **MediaPipe Hands JS** in the browser (client-side)
- Extracts **21 hand landmarks → 63 floats**
- Streams landmarks to a FastAPI backend using **WebSocket**
- Backend predicts **letter/word** in **English (ASL)** and **Arabic (ArSL)**
- UI shows:
  - Camera overlay (landmarks)
  - Mode badge (letter/word/auto)
  - Prediction + confidence
  - Built text/sentence
  - English/Arabic toggle (RTL support)

**Important rule:** Your **model architectures remain the same**. Deployment changes only packaging + stable APIs.

---

## 2) Recommended architecture (clean + scalable)

### Frontend (Vercel)
- React + Vite + TypeScript
- Tailwind CSS (modern dark style)
- MediaPipe Hands JS runs in browser
- WebSocket to backend for real-time predictions

### Backend (Railway)
- FastAPI
- WebSocket endpoint: `/ws/recognize`
- Optional REST endpoints: `/health`, `/predict/letter`, `/predict/word`
- Loads models once on startup (per language)

### Data flow
Browser webcam → MediaPipe → 63 floats/frame → WS → FastAPI → inference → WS response → UI

---

## 3) Project structure (deployment-ready)

```
SLR-Main/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── predict.py
│   │   │   └── websocket.py
│   │   └── models/
│   │       ├── loader.py
│   │       ├── preprocess.py
│   │       ├── letter_predictor.py
│   │       ├── word_predictor.py
│   │       ├── mode_detector.py
│   │       ├── letter_decoder.py
│   │       └── word_decoder.py
│   ├── model_files/
│   │   ├── letters/
│   │   └── words/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── railway.toml
│
├── web/
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vercel.json
│   ├── .env.example
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── pages/
│       │   ├── Home.tsx
│       │   └── Recognize.tsx
│       ├── components/
│       │   ├── TopBar.tsx
│       │   ├── CameraPanel.tsx
│       │   ├── PredictionCard.tsx
│       │   ├── OutputCard.tsx
│       │   └── ControlsCard.tsx
│       ├── hooks/
│       │   ├── useMediaPipe.ts
│       │   └── useWebSocket.ts
│       ├── types/
│       │   └── index.ts
│       └── utils/
│           ├── constants.ts
│           └── landmarks.ts
│
└── Deployment/
    ├── docs/
    └── scripts/
```

---

## 4) Modern dark UI design system (consistent look)

### Colors
- Page background: `#0b1220`
- Card: `#111827`
- Border: `#1f2937`
- Primary: Indigo `#6366f1`
- Word-mode accent: Purple `#a855f7`
- Success: Emerald `#10b981`

### Card container (reusable)
Use this style for all containers/cards:
- `bg-gray-900 border border-gray-800 rounded-2xl shadow-lg shadow-black/20`

Text:
- Title: `text-xs uppercase tracking-widest text-gray-400`
- Body: `text-gray-100`
- Secondary: `text-gray-400 text-sm`

Arabic support:
- Output container uses: `dir="rtl"` + `text-right` when Arabic.

---

## 5) Stable API contract (frontend ↔ backend)

### WebSocket endpoint
- `WS /ws/recognize`

Client → server (per frame):
```json
{ "landmarks": [63 floats], "language": "en" }
```

Optional command:
```json
{ "command": "clear" }
```

Server → client:
```json
{
  "mode": "letter" | "word",
  "prediction": {
    "label": "A",
    "confidence": 0.92,
    "top": [{ "label": "B", "confidence": 0.04 }]
  },
  "decoder": {
    "text": "HELLO",
    "sentence": "I NEED HELP"
  },
  "frames_buffered": 12,
  "frames_needed": 30
}
```

---

## 6) Backend approach (build safely)

### Phase B0 — Dummy backend first (fast progress)
- Implement WS endpoint returning fake predictions.
- This lets frontend + design + streaming work before model integration.

### Phase B1 — Add real model registry
- Place all deployment artifacts in: `backend/model_files/`
- Load them on startup into a registry:
  - `registry["en"]["word"]`
  - `registry["ar"]["word"]`

### Phase B2 — Match preprocessing
- Use same landmark normalization as training
- Apply scaler stats if you used scaling during training
- Word model uses a fixed buffer (e.g., 30 frames)

---

## 7) Frontend approach

### Phase F0 — UI first with mock data
- Build all containers and layout with hardcoded sample JSON.
- Confirm modern style, responsiveness, RTL.

### Phase F1 — Add MediaPipe hands
- Extract landmarks
- Convert to 63 floats (relative-to-wrist is recommended)
- Validate landmarks (non-zero)

### Phase F2 — Add WebSocket streaming
- Connect to `VITE_WS_URL`
- Send landmarks every frame (or every 2–3 frames)
- Render prediction + decoder

---

## 8) Local run plan (must work before cloud)

### Backend local
```bash
cd backend
python -m venv .venv
# activate venv
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Web local
```bash
cd web
npm install
npm run dev
```

`web/.env`:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/recognize
```

---

## 9) Cloud deployment plan

### Backend → Railway
- Deploy from GitHub
- Run uvicorn with `$PORT`
- Set CORS allowed origins to include your Vercel domain
- WebSocket URL becomes:
  - `wss://<railway-domain>/ws/recognize`

### Frontend → Vercel
- Root dir: `web`
- Build: `npm run build`
- Output: `dist`
- Env vars:
```env
VITE_API_URL=https://<railway-domain>
VITE_WS_URL=wss://<railway-domain>/ws/recognize
```

**Critical:** Use `wss://` in production.

---

## 10) Execution order (best path to finish)
1. Frontend UI containers (modern dark) with mock JSON
2. Backend WS (dummy responses)
3. Connect WS end-to-end locally
4. Add MediaPipe → send real landmarks
5. Load real models in backend
6. Add buffering + decoders
7. Deploy backend (Railway)
8. Deploy frontend (Vercel)
9. Final demo polish + fallback recording

---

## 11) Tools you need (software + accounts)

### A) Local development tools
- **Git** (version control)
- **Python 3.10+** (backend + scripts)
- **Node.js 20 LTS** + npm (web frontend)
- **VS Code** recommended extensions:
  - Python
  - Pylance
  - ESLint
  - Prettier
  - Tailwind CSS IntelliSense
  - React/TypeScript tooling
- **Google Chrome** (best for webcam + debugging)

### B) Python / Backend dependencies (typical)
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `numpy`
- `pandas` (optional, for reading classes CSVs)
- `tensorflow` (or `tensorflow-cpu` for Railway, depending on plan)
- `scikit-learn` (if you need scaler/encoder utilities)
- `python-multipart` (if you later add file uploads)
- `websockets` (usually bundled via uvicorn standard)

### C) Web / Frontend dependencies
- React + Vite + TypeScript
- Tailwind CSS
- `react-router-dom`
- MediaPipe packages:
  - `@mediapipe/hands`
  - `@mediapipe/camera_utils`
  - `@mediapipe/drawing_utils`
- (Optional) i18n:
  - `i18next`
  - `react-i18next`

### D) Cloud accounts
- **GitHub** (repo hosting)
- **Railway** (backend deploy)
- **Vercel** (frontend deploy)

### E) Optional but helpful tools
- **Postman** (REST testing)
- **websocat** (WebSocket testing) or a simple python WS client
- **Docker Desktop** (if you want to test container locally)

---

## 12) Notes for graduation reliability
- Build dummy mode first (demo always works)
- Add a “fallback demo” recording (screen capture) in case network fails
- Keep models small enough for cloud CPU or reduce inference rate
- Ensure preprocessing is identical between training + frontend/backend