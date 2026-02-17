# 10 — Testing & Verification Checklist

> Step-by-step checklist to verify everything works correctly.

---

## Phase 1: Model Files Verification

| # | Check | How | Expected | ☐ |
|---|-------|-----|----------|---|
| 1 | Model files exist | `ls backend/model_files/` | 4 .h5 files, 4 .csv files | ☐ |
| 2 | ASL Letter model loads | Python: `tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")` | No errors, input (None, 63) | ☐ |
| 3 | ArSL Letter model loads | Same as above with ArSL model | No errors, input (None, 63) | ☐ |
| 4 | Word model loads | Load with `custom_objects={'TemporalAttention': TemporalAttention}` | No errors, input (None, 30, 63) | ☐ |
| 5 | ASL labels load | `pd.read_csv("asl_mediapipe_keypoints_dataset.csv")['label'].nunique()` | 29 classes | ☐ |
| 6 | ArSL labels load | Same with ArSL CSV | Arabic letter classes | ☐ |
| 7 | Word classes load | `pd.read_csv("asl_word_classes.csv")` | 158 rows | ☐ |
| 8 | Word vocabulary loads | `pd.read_csv("shared_word_vocabulary.csv")` | 157 rows, has english + arabic | ☐ |

---

## Phase 2: Backend API Tests

### Start the server:
```powershell
cd "m:\Term 10\Grad\Deployment\backend"
.\venv\Scripts\Activate
python -m uvicorn app.main:app --reload --port 8000
```

| # | Test | Method | Expected | ☐ |
|---|------|--------|----------|---|
| 1 | Root endpoint | GET `http://localhost:8000/` | JSON with app name and endpoints | ☐ |
| 2 | Health check | GET `http://localhost:8000/health` | `models_loaded: true`, all 3 models true | ☐ |
| 3 | Swagger docs | Browser `http://localhost:8000/docs` | Interactive API documentation | ☐ |
| 4 | Letter prediction | POST `/predict/letter` with 63 floats | Returns letter + confidence | ☐ |
| 5 | Word prediction | POST `/predict/word` with 30×63 floats | Returns word + confidence | ☐ |
| 6 | Arabic letter | POST `/predict/letter` with `language: "ar"` | Returns Arabic letter | ☐ |
| 7 | Invalid input | POST `/predict/letter` with 10 floats | Returns 422 error | ☐ |
| 8 | WebSocket | Connect to `ws://localhost:8000/ws/recognize` | Connection accepted | ☐ |

### Quick Test Script:

```python
"""Quick test — run while backend is running."""
import requests
import numpy as np
import json

BASE = "http://localhost:8000"

# Test 1: Health
r = requests.get(f"{BASE}/health")
print(f"Health: {r.json()['status']}")
assert r.json()['models_loaded'] == True

# Test 2: Letter prediction (random landmarks)
landmarks = np.random.randn(63).tolist()
r = requests.post(f"{BASE}/predict/letter", json={
    "landmarks": landmarks,
    "language": "en"
})
print(f"Letter: {r.json()['letter']} ({r.json()['confidence']:.2f})")

# Test 3: Word prediction (random frames)
frames = np.random.randn(30, 63).tolist()
r = requests.post(f"{BASE}/predict/word", json={
    "frames": frames,
    "language": "en"
})
print(f"Word: {r.json()['word_en']} ({r.json()['confidence']:.2f})")

# Test 4: Arabic
r = requests.post(f"{BASE}/predict/letter", json={
    "landmarks": landmarks,
    "language": "ar"
})
print(f"Arabic Letter: {r.json()['letter']} ({r.json()['confidence']:.2f})")

print("\nAll tests passed! ✓")
```

### WebSocket Test:

```python
"""WebSocket test — run while backend is running."""
import asyncio
import json
import websockets
import numpy as np

async def test_ws():
    async with websockets.connect("ws://localhost:8000/ws/recognize") as ws:
        # Send 5 frames
        for i in range(5):
            landmarks = np.random.randn(63).tolist()
            await ws.send(json.dumps({
                "landmarks": landmarks,
                "language": "en"
            }))
            response = await ws.recv()
            data = json.loads(response)
            print(f"Frame {i+1}: mode={data['mode']}, prediction={data.get('prediction', {}).get('letter', 'N/A')}")

        # Test clear command
        await ws.send(json.dumps({"command": "clear"}))
        response = await ws.recv()
        print(f"Clear: {json.loads(response)}")

    print("\nWebSocket test passed! ✓")

asyncio.run(test_ws())
```

---

## Phase 3: Web Frontend Tests

### Start the web app:
```powershell
cd "m:\Term 10\Grad\Deployment\web"
npm run dev
# Opens at http://localhost:5173
```

| # | Test | How | Expected | ☐ |
|---|------|-----|----------|---|
| 1 | Home page loads | Visit `localhost:5173` | ESHARA branding, "Start Recognition" button | ☐ |
| 2 | Navigate to recognize | Click "Start Recognition" | Camera feed appears | ☐ |
| 3 | Camera permission | Browser asks for camera | Grant permission → video shows | ☐ |
| 4 | MediaPipe loads | Wait 2-3 seconds | "Loading MediaPipe..." disappears | ☐ |
| 5 | Hand detection | Show hand to camera | Green landmarks drawn on hand | ☐ |
| 6 | WebSocket connects | Check badge | "Connected" badge appears (green) | ☐ |
| 7 | Letter mode works | Hold a letter sign still | Prediction shows with confidence | ☐ |
| 8 | Text builds | Sign multiple letters | Letters accumulate in text box | ☐ |
| 9 | Word mode triggers | Move hand in sign motion | Mode badge changes to "Word Mode" | ☐ |
| 10 | Language toggle | Click "العربية" | Labels switch to Arabic, predictions too | ☐ |
| 11 | Clear button | Click "Clear" | Text box clears, state resets | ☐ |
| 12 | Stop/Start | Click "Stop" then "Start" | Camera stops and restarts | ☐ |
| 13 | No hand shown | Remove hand from view | "No Hand" badge appears | ☐ |
| 14 | RTL text direction | Switch to Arabic | Text box aligns right-to-left | ☐ |

---

## Phase 4: Mobile App Tests

| # | Test | How | Expected | ☐ |
|---|------|-----|----------|---|
| 1 | App opens | `npx expo start` → scan QR | Home screen appears | ☐ |
| 2 | Models load | Navigate to recognize | "Loading models..." then camera | ☐ |
| 3 | Camera works | Grant permission | Front camera shows | ☐ |
| 4 | Hand detected | Show hand to camera | Detection works | ☐ |
| 5 | Letter prediction | Hold letter sign | Prediction + confidence shown | ☐ |
| 6 | Word prediction | Perform word sign | Word predicted | ☐ |
| 7 | Text builds | Sign multiple letters | Text accumulates | ☐ |
| 8 | Language switch | Tap language button | Switches EN ↔ AR | ☐ |
| 9 | Mode switch | Tap mode button | Switches letter ↔ word | ☐ |
| 10 | Offline test | Turn off WiFi | Still works (on-device) | ☐ |
| 11 | Performance | Normal use | Smooth camera, no lag | ☐ |
| 12 | Clear | Tap Clear | All state resets | ☐ |

---

## Phase 5: TFLite Model Tests

| # | Test | How | Expected | ☐ |
|---|------|-----|----------|---|
| 1 | ASL letter TFLite loads | `tf.lite.Interpreter(asl_letter_model.tflite)` | No errors | ☐ |
| 2 | ASL letter TFLite predicts | Random (1,63) input | 29-class output, sums to ~1.0 | ☐ |
| 3 | ArSL letter TFLite loads | Same with ArSL model | No errors | ☐ |
| 4 | ArSL letter TFLite predicts | Random (1,63) input | Correct class count output | ☐ |
| 5 | Word TFLite loads | May need Flex delegate | No errors | ☐ |
| 6 | Word TFLite predicts | Random (1,30,63) input | 157-class output, sums to ~1.0 | ☐ |
| 7 | Label JSON files valid | `json.load()` each file | Parsed correctly, right count | ☐ |

---

## Phase 6: Cloud Deployment Tests

| # | Test | Target URL | Expected | ☐ |
|---|------|-----------|----------|---|
| 1 | Railway health | `https://YOUR-APP.up.railway.app/health` | `status: healthy` | ☐ |
| 2 | Railway docs | `https://YOUR-APP.up.railway.app/docs` | Swagger UI loads | ☐ |
| 3 | Railway letter API | POST `/predict/letter` to Railway URL | Prediction returned | ☐ |
| 4 | Railway WebSocket | Connect `wss://YOUR-APP.up.railway.app/ws/recognize` | Connection works | ☐ |
| 5 | Vercel loads | `https://eshara.vercel.app` | Home page shows | ☐ |
| 6 | Vercel camera | Navigate to recognize | Camera + MediaPipe works | ☐ |
| 7 | Vercel → Railway | Show sign on Vercel app | Prediction appears (via Railway) | ☐ |
| 8 | HTTPS working | Check browser padlock | Green padlock on both URLs | ☐ |
| 9 | CORS correct | No console errors | No CORS blocked requests | ☐ |

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "Model not found" | Files not copied | Run `python scripts/copy_models.py` |
| "TemporalAttention not found" | Custom layer not registered | Add `custom_objects` when loading |
| "CORS error" in browser | Backend not allowing frontend origin | Update `CORS_ORIGINS` env var |
| WebSocket won't connect | Wrong URL or server not running | Check URL is `ws://` (local) or `wss://` (prod) |
| Camera permission denied | Browser blocked camera | Check browser settings → Allow camera |
| MediaPipe slow to load | First load downloads ~5MB of model files | Normal, cached after first load |
| CUDA errors on Railway | GPU not available on cloud | OK — CPU inference works fine, just slower |
| TFLite "Flex delegate" error | Word model needs SELECT_TF_OPS | Install TFLite with FlexDelegate on mobile |
| Railway 503 error | Cold start, model loading | Wait 30s and retry; models are loading |
| Build fails on Railway | Not enough memory | Reduce Docker image size or upgrade plan |
