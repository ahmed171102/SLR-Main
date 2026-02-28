# ESHARA — Separated Pipelines

Bilingual sign-language recognition system with **fully independent English (ASL)
and Arabic (ArSL) pipelines**, a confidence-gated **LLM correction agent**
(GPT-4o-mini), and a **FastAPI** backend for REST + WebSocket real-time inference.

---

## Architecture Overview

```
Camera Frame
     │
     ├──► /ws/en  (English Pipeline)
     │        MediaPipe → ModeDetector → LetterPredictor / WordPredictor
     │        → LetterDecoder / WordDecoder → LLM Correction → JSON result
     │
     └──► /ws/ar  (Arabic Pipeline)
              MediaPipe → ModeDetector → ArabicLetterPredictor / ArabicWordPredictor
              → ArabicLetterDecoder / ArabicWordDecoder → LLM Correction → JSON result
```

Each pipeline has its **own MediaPipe instance, models, decoders, and LLM prompts**.
There is zero cross-contamination between languages.

---

## Folder Structure

```
Separated Pipelines/
├── config/
│   └── settings.py          # All thresholds, paths, LLM config
├── shared/
│   ├── utils/
│   │   ├── mediapipe_extractor.py
│   │   └── mode_detector.py
│   └── models/
│       ├── temporal_attention.py   # Custom Keras layer for word BiLSTM
│       └── model_loader.py         # load_keras_model, load_label_encoder, etc.
├── llm_agent/
│   ├── prompts.py            # EN + AR system prompts for letter/word correction
│   └── correction_agent.py   # LLMCorrectionAgent with confidence gating + cache
├── english_pipeline/
│   ├── models/
│   │   ├── letter_predictor.py
│   │   └── word_predictor.py
│   ├── decoders/
│   │   ├── letter_decoder.py
│   │   └── word_decoder.py
│   └── inference/
│       └── pipeline.py       # EnglishPipeline.process_frame()
├── arabic_pipeline/
│   ├── models/
│   │   ├── letter_predictor.py   # NAME_TO_ARABIC mapping
│   │   └── word_predictor.py     # Graceful fallback if KArSL model missing
│   ├── decoders/
│   │   ├── letter_decoder.py
│   │   └── word_decoder.py
│   └── inference/
│       └── pipeline.py       # ArabicPipeline.process_frame()
├── backend/
│   └── app/
│       ├── main.py           # FastAPI app with lifespan (model loading)
│       ├── schemas.py        # Pydantic request/response models
│       └── routes/
│           ├── health.py     # GET /health, GET /info
│           ├── predict.py    # POST /api/predict, POST /api/predict/base64
│           └── websocket.py  # WS /ws/{lang}
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd "SLR Main/Separated Pipelines"
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file (or export directly):

```env
OPENAI_API_KEY=sk-...        # For LLM correction (optional)
```

### 3. Ensure Model Files Exist

The pipelines reference trained model files from the parent project.
Update paths in `config/settings.py` if your directory layout differs:

| Model | Expected Path (relative to SLR Main) |
|-------|--------------------------------------|
| ASL Letter MLP | `Letters/ASL Letter (English)/Models/asl_mediapipe_mlp_model.h5` |
| ASL Letter Encoder | `Letters/ASL Letter (English)/Models/label_encoder_classes.csv` |
| ArSL Letter MLP | `Letters/ArSL Letter (Arabic)/Models/arsl_mediapipe_mlp_model.h5` |
| ArSL Letter Encoder | `Letters/ArSL Letter (Arabic)/Models/arsl_label_encoder_classes.csv` |
| ASL Word BiLSTM | `Words/ASL Word (English)/Models/best_bilstm_model.h5` |
| ASL Word Classes | `Words/ASL Word (English)/Models/word_classes.npy` |
| ArSL Word BiLSTM | `Words/ArSL Word (Arabic)/Models/best_bilstm_model.h5` |
| ArSL Word Classes | `Words/ArSL Word (Arabic)/Models/word_classes.npy` |

### 4. Run the Server

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test

- Health check: `GET http://localhost:8000/health`
- Info: `GET http://localhost:8000/info`
- Predict (form): `POST http://localhost:8000/api/predict` with `lang=en` and `frame=<file>`
- WebSocket: connect to `ws://localhost:8000/ws/en` or `ws://localhost:8000/ws/ar`

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Pipeline status |
| `GET` | `/info` | Project metadata and available endpoints |
| `POST` | `/api/predict` | Predict from uploaded image (form-data: `lang`, `frame`) |
| `POST` | `/api/predict/base64` | Predict from JSON `{"lang": "en", "image": "<b64>"}` |
| `POST` | `/api/reset/{lang}` | Reset decoder state for `en` or `ar` |

### WebSocket

| Path | Description |
|------|-------------|
| `ws://host:port/ws/en` | Real-time English ASL inference |
| `ws://host:port/ws/ar` | Real-time Arabic ArSL inference |

**Protocol:**
- **Binary message** → raw JPEG/PNG bytes (lowest latency)
- **Text message** → `{"image": "<base64>"}` or `{"command": "reset"}`
- **Server response** → JSON matching `PredictionResponse` schema

---

## Key Design Decisions

1. **Separate pipelines**: Each language has its own MediaPipe instance,
   models, decoders, and LLM prompts. No shared mutable state.

2. **Mode detection**: Motion-based (landmark velocity). Still hand → letter
   spelling, moving hand → word-level recognition.

3. **LLM correction**: GPT-4o-mini with confidence gating (threshold 0.75).
   Only triggered when model confidence is below the gate. LRU cache (500
   entries) prevents redundant API calls.

4. **Graceful degradation**: Arabic word model may be missing (KArSL-502 not
   yet available). `ArabicWordPredictor` returns a placeholder instead of
   crashing.

---

## Models Summary

| Pipeline | Task | Architecture | Classes | Notes |
|----------|------|-------------|---------|-------|
| English | Letters | MLP (MediaPipe) | 29 | A-Z + space/del/nothing |
| English | Words | BiLSTM + Attention | 157 | 30-frame sequences |
| Arabic | Letters | MLP (MediaPipe) | 31 | 28 Arabic letters + controls |
| Arabic | Words | BiLSTM + Attention | TBD | Needs KArSL-502 dataset |

---

## License

See the parent project's LICENSE file.
