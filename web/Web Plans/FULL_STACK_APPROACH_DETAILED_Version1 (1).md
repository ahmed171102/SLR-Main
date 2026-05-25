# Full-Stack Approach (Detailed) — Graduation Project SLR (Modern Web + API)

**Date:** 2026-04-14  
**Goal:** Build and deploy a modern full-stack app (frontend + backend + cloud) that supports **real-time webcam recognition UI** with **English/Arabic toggle**, using **MediaPipe in the browser** and a **FastAPI backend** via **WebSocket**.  
> This document focuses on the **full-stack engineering approach** (structure, linking, deployment, quality), not training.

---

## 1) End Product (What you will demo)

### User experience
- User opens the website (Vercel)
- Clicks “Start Recognition”
- Browser asks for webcam permission
- Hand landmarks appear on screen (overlay)
- Live prediction appears with confidence bar
- Output text builds over time
- User toggles language: English ↔ Arabic (RTL)

### Technical experience
- Frontend streams landmark data to backend over WebSocket
- Backend returns standardized JSON responses
- Works locally and in production with secure WebSockets (`wss://`)

---

## 2) Architecture (the clean, modern pattern)

### 2.1 Components
**Frontend (Vercel)**
- React + Vite + TypeScript
- Tailwind modern dark theme
- MediaPipe Hands JS (client-side)
- WebSocket client to backend

**Backend (Railway)**
- FastAPI
- WebSocket endpoint `/ws/recognize`
- REST endpoint `/health` for monitoring/debugging
- CORS for local + production domains

### 2.2 Data Flow
Browser Webcam  
→ MediaPipe Hands (JS)  
→ landmarks array (63 floats per frame)  
→ WebSocket send  
→ FastAPI receive  
→ (prediction logic called here)  
→ JSON response  
→ UI update (prediction + sentence)

---

## 3) Repository & Folder Structure (Production-ready)

### 3.1 Top-level structure
```
SLR-Main/
├── backend/
├── web/
└── Deployment/
```

### 3.2 Backend structure (FastAPI)
```
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── routes/
│   │   ├── health.py
│   │   ├── predict.py          # optional REST
│   │   └── websocket.py        # main WS endpoint
│   └── services/
│       ├── predictor.py        # ONE entry point for prediction
│       ├── decoder.py          # build text/sentence over time (optional)
│       └── validators.py       # landmark validation
├── requirements.txt
├── Dockerfile
└── README.md
```

**Design principle:** Keep a single service function `predictor.predict(payload)` so swapping “dummy prediction” → “real prediction” is one change.

### 3.3 Frontend structure (React)
```
web/
├── package.json
├── tailwind.config.js
├── vercel.json
├── .env.example
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── pages/
    │   ├── Home.tsx
    │   └── Recognize.tsx
    ├── components/
    │   ├── TopBar.tsx
    │   ├── CameraPanel.tsx
    │   ├── ControlsCard.tsx
    │   ├── PredictionCard.tsx
    │   └── OutputCard.tsx
    ├── hooks/
    │   ├── useMediaPipe.ts
    │   └── useWebSocket.ts
    ├── types/
    │   └── index.ts
    └── utils/
        ├── constants.ts
        └── landmarks.ts
```

---

## 4) Communication Contract (the most important part)

### 4.1 Environment variables (never hardcode URLs)
**Local (`web/.env`)**
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/recognize
```

**Production (Vercel env vars)**
```env
VITE_API_URL=https://<railway-domain>
VITE_WS_URL=wss://<railway-domain>/ws/recognize
```

### 4.2 WebSocket payload contract
Frontend → backend:
```json
{
  "landmarks": [63 floats],
  "language": "en"
}
```

Optional commands:
```json
{ "command": "clear" }
{ "command": "ping" }
```

Backend → frontend:
```json
{
  "mode": "letter",
  "prediction": {
    "label": "SAMPLE",
    "confidence": 0.99,
    "top": [
      { "label": "SAMPLE", "confidence": 0.99 }
    ]
  },
  "decoder": {
    "text": "",
    "sentence": ""
  },
  "frames_buffered": 0,
  "frames_needed": 30,
  "error": null
}
```

**Rule:** Keep the response shape stable so the UI is simple and reliable.

---

## 5) Backend Approach (FastAPI) — Step-by-step

### Phase B0 — Build backend skeleton (day 1)
1. Create `GET /health`
2. Create `WS /ws/recognize`
3. Add CORS for:
   - `http://localhost:5173`
   - your Vercel URL

4. Implement a dummy predictor:
- returns `"SAMPLE"` with confidence `0.9`

**Acceptance:** You can connect from the frontend and receive responses.

### Phase B1 — Add logging & validation (day 1–2)
Add:
- request validation: landmarks length must be 63
- handle JSON parse errors
- handle disconnects gracefully
- structured logs:
  - connect/disconnect
  - message rate
  - errors

**Acceptance:** Backend never crashes from bad payloads.

### Phase B2 — Add per-connection state (day 2–3)
Implement state object per WebSocket connection:
- `language`
- `mode`
- optional buffers for smoothing / word window
- “clear” resets state

**Acceptance:** Clear button resets output correctly.

---

## 6) Frontend Approach (React + MediaPipe) — Step-by-step

### Phase F0 — Create UI with mock data (day 1)
Build modern dark containers:
- Camera panel (placeholder)
- Prediction card
- Output card
- Controls card
- EN/AR toggle

**Acceptance:** UI looks professional and responsive.

### Phase F1 — Implement WebSocket hook (day 1–2)
`useWebSocket.ts`:
- connect to `VITE_WS_URL`
- expose:
  - `isConnected`
  - `lastResponse`
  - `sendLandmarks(landmarks, language)`
  - `sendCommand("clear")`

**Acceptance:** UI shows “connected” and receives JSON.

### Phase F2 — Implement MediaPipe hook (day 2–3)
`useMediaPipe.ts`:
- start camera
- run MediaPipe Hands
- extract landmarks and convert to float[63]
- call `sendLandmarks(...)`

**Acceptance:** Backend logs show landmarks coming in.

### Phase F3 — RTL & language UX (day 3)
- Language toggle:
  - send `language` with each message
  - output container uses `dir="rtl"` when `ar`
- change labels in UI:
  - “English/العربية”
  - text alignment and fonts

**Acceptance:** Arabic display is readable and correct direction.

---

## 7) Modern Dark UI Design (graduation-ready)

### Page layout (Recognize)
- Desktop:
  - left 2/3: camera container
  - right 1/3: controls + prediction + output
- Mobile:
  - camera first, then controls, then output

### Reusable card style
- `bg-gray-900 border border-gray-800 rounded-2xl shadow-lg shadow-black/20`
- Title: `text-xs uppercase tracking-widest text-gray-400`

### Status badges
- connected: green
- disconnected: red
- mode: indigo/purple
- hand detected: green/amber

---

## 8) Local Development & Testing (must do)

### Backend local run
```bash
cd backend
python -m venv .venv
# activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Web local run
```bash
cd web
npm install
npm run dev
```

### Manual checks
- `http://localhost:8000/health` returns OK
- Web shows “connected”
- Webcam opens
- Predictions update (even dummy)

---

## 9) Cloud Deployment (Railway + Vercel) — Step-by-step

### 9.1 Deploy backend to Railway
1. Push your repo to GitHub
2. Railway → New Project → Deploy from GitHub
3. Ensure start command uses `$PORT`:
   - `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add env var:
   - `ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app`
5. Verify:
   - `https://<railway-domain>/health` works

### 9.2 Deploy frontend to Vercel
1. Vercel → Import repo
2. Root directory: `web`
3. Build: `npm run build`
4. Output: `dist`
5. Env vars:
   - `VITE_API_URL=https://<railway-domain>`
   - `VITE_WS_URL=wss://<railway-domain>/ws/recognize`
6. Deploy
7. Verify:
   - web loads
   - WS connects
   - predictions appear

---

## 10) Common deployment issues (and fixes)

### Issue: WS works locally but fails in production
Cause: `ws://` used instead of `wss://`
Fix: set `VITE_WS_URL` to `wss://...`

### Issue: CORS blocked
Cause: backend doesn’t allow your Vercel domain
Fix: add Vercel domain to `ALLOWED_ORIGINS`

### Issue: Backend 404 on `/ws/recognize`
Cause: route mismatch
Fix: ensure exact path `/ws/recognize`

---

## 11) Recommended execution timeline (fastest path)

### Day 1
- Build backend skeleton + dummy WS response
- Build frontend UI + connect WS

### Day 2
- Add MediaPipe in web
- Stream landmarks to backend

### Day 3
- RTL polish + language toggle
- Cloud deploy Railway + Vercel

### Day 4–5
- Polish UI, stability, fallback demo recording

---

## 12) Tools Required (exact list)

### Local tools
- Git
- Python 3.10+ (recommended 3.10/3.11)
- Node.js 20 LTS + npm
- VS Code (recommended)

### Backend packages
- fastapi
- uvicorn[standard]
- pydantic
- numpy (for arrays/validation)

### Frontend packages
- react / react-dom
- vite
- typescript
- tailwindcss
- react-router-dom
- @mediapipe/hands
- @mediapipe/camera_utils
- @mediapipe/drawing_utils

### Cloud accounts
- GitHub
- Railway
- Vercel

---

## 13) Acceptance checklist (you are ready to present)

- [ ] Backend `/health` works locally + on Railway
- [ ] Web loads locally + on Vercel
- [ ] WebSocket connects locally + on Vercel (`wss://`)
- [ ] Webcam opens and landmarks render
- [](#)
