# 05 — Backend API: Step-by-Step Build Guide

> Build the Python FastAPI backend that loads your ML models and serves predictions.

---

## Overview

The backend is a Python API server that:
1. Loads your trained .h5 models on startup
2. Accepts hand landmark data (63 floats per frame) from web/mobile
3. Runs predictions through the correct model (letter or word)
4. Returns the predicted sign + confidence
5. Supports both REST (single request) and WebSocket (real-time streaming)

```
Client sends: [63 floats] → Backend → Model predicts → Returns: {"prediction": "A", "confidence": 0.95}
```

---

## Prerequisites

- Python 3.9 installed
- Model files copied to `backend/model_files/` (run `scripts/copy_models.py` first)
- Terminal / command prompt

---

## Step 1: Create Virtual Environment

```powershell
cd "m:\Term 10\Grad\Deployment\backend"

# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\Activate

# You should see (venv) in your terminal prompt
```

---

## Step 2: Install Dependencies

Create `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
tensorflow==2.10.0
mediapipe==0.10.8
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.3.2
python-dotenv==1.0.0
websockets==12.0
pydantic==2.5.2
python-multipart==0.0.6
```

Install:
```powershell
pip install -r requirements.txt
```

---

## Step 3: Configuration File

### `app/config.py`

```python
"""Application configuration — loads from environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_files"

# Model files
ASL_LETTER_MODEL = MODEL_DIR / "asl_mediapipe_mlp_model.h5"
ARSL_LETTER_MODEL = MODEL_DIR / "arsl_mediapipe_mlp_model_final.h5"
ASL_WORD_MODEL = MODEL_DIR / "asl_word_lstm_model_best.h5"

# Label files
ASL_LETTER_LABELS_CSV = MODEL_DIR / "asl_mediapipe_keypoints_dataset.csv"
ARSL_LETTER_LABELS_CSV = MODEL_DIR / "FINAL_CLEAN_DATASET.csv"
WORD_CLASSES_CSV = MODEL_DIR / "asl_word_classes.csv"
WORD_VOCABULARY_CSV = MODEL_DIR / "shared_word_vocabulary.csv"

# Model parameters
SEQUENCE_LENGTH = 30          # Frames for word model
NUM_FEATURES = 63             # 21 landmarks × 3 coords

# Prediction thresholds
LETTER_CONFIDENCE_THRESHOLD = 0.85
WORD_CONFIDENCE_THRESHOLD = 0.35

# Letter decoder settings
LETTER_STABLE_WINDOW = 5      # Frames before committing letter
LETTER_MAJORITY_RATIO = 0.7   # Required agreement ratio
LETTER_COOLDOWN = 0.6          # Seconds between same letter

# Word decoder settings
WORD_STABILITY_WINDOW = 3     # Predictions before committing word
WORD_COOLDOWN = 2.0            # Seconds between words
WORD_PREDICTION_INTERVAL = 0.5 # Seconds between predictions

# Mode detection
MOTION_THRESHOLD_LETTER = 0.015  # Below = still hand = letter mode
MOTION_THRESHOLD_WORD = 0.03     # Above = moving hand = word mode

# Server
PORT = int(os.getenv("BACKEND_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
```

---

## Step 4: Custom Keras Layer

### `app/models/temporal_attention.py`

This MUST exist before loading the word model:

```python
"""TemporalAttention — custom Keras layer used in word BiLSTM model."""
import tensorflow as tf

class TemporalAttention(tf.keras.layers.Layer):
    """Attention mechanism over temporal (sequence) dimension."""

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()
```

---

## Step 5: Model Loader

### `app/models/loader.py`

```python
"""Load all ML models and label encoders on startup."""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from app.config import (
    ASL_LETTER_MODEL, ARSL_LETTER_MODEL, ASL_WORD_MODEL,
    ASL_LETTER_LABELS_CSV, ARSL_LETTER_LABELS_CSV,
    WORD_CLASSES_CSV, WORD_VOCABULARY_CSV
)
from app.models.temporal_attention import TemporalAttention


class ModelManager:
    """Singleton that holds all loaded models and label mappings."""

    def __init__(self):
        self.asl_letter_model = None
        self.arsl_letter_model = None
        self.asl_word_model = None
        self.asl_letter_encoder = None    # LabelEncoder
        self.arsl_letter_encoder = None   # LabelEncoder
        self.word_classes = None          # {model_index: word_id}
        self.word_vocabulary = None       # {word_id: {english, arabic, category}}
        self._loaded = False

    def load_all(self):
        """Load all models and label mappings. Call once on startup."""
        if self._loaded:
            return

        print("Loading models...")

        # 1. ASL Letter MLP
        self.asl_letter_model = tf.keras.models.load_model(
            str(ASL_LETTER_MODEL)
        )
        print(f"  ✓ ASL Letter MLP: {ASL_LETTER_MODEL.name}")

        # 2. ArSL Letter MLP
        self.arsl_letter_model = tf.keras.models.load_model(
            str(ARSL_LETTER_MODEL)
        )
        print(f"  ✓ ArSL Letter MLP: {ARSL_LETTER_MODEL.name}")

        # 3. ASL Word BiLSTM (needs TemporalAttention)
        self.asl_word_model = tf.keras.models.load_model(
            str(ASL_WORD_MODEL),
            custom_objects={'TemporalAttention': TemporalAttention}
        )
        print(f"  ✓ ASL Word BiLSTM: {ASL_WORD_MODEL.name}")

        # 4. ASL Letter LabelEncoder
        df = pd.read_csv(str(ASL_LETTER_LABELS_CSV))
        self.asl_letter_encoder = LabelEncoder()
        self.asl_letter_encoder.fit(df['label'])
        print(f"  ✓ ASL Letter Labels: {len(self.asl_letter_encoder.classes_)} classes")

        # 5. ArSL Letter LabelEncoder
        df = pd.read_csv(str(ARSL_LETTER_LABELS_CSV))
        self.arsl_letter_encoder = LabelEncoder()
        self.arsl_letter_encoder.fit(df['label'])
        print(f"  ✓ ArSL Letter Labels: {len(self.arsl_letter_encoder.classes_)} classes")

        # 6. Word class mappings
        word_df = pd.read_csv(str(WORD_CLASSES_CSV))
        self.word_classes = dict(zip(
            word_df['model_class_index'].astype(int),
            word_df['word_id'].astype(int)
        ))
        print(f"  ✓ Word Classes: {len(self.word_classes)} mappings")

        # 7. Word vocabulary (bilingual)
        vocab_df = pd.read_csv(str(WORD_VOCABULARY_CSV))
        self.word_vocabulary = {}
        for _, row in vocab_df.iterrows():
            self.word_vocabulary[int(row['word_id'])] = {
                'english': row['english'],
                'arabic': row['arabic'],
                'category': row.get('category', 'unknown')
            }
        print(f"  ✓ Word Vocabulary: {len(self.word_vocabulary)} words")

        self._loaded = True
        print("All models loaded successfully!")


# Global singleton
model_manager = ModelManager()
```

---

## Step 6: Letter Predictor

### `app/models/letter_predictor.py`

```python
"""Single-frame letter prediction using MLP model."""
import numpy as np
from app.models.loader import model_manager
from app.config import LETTER_CONFIDENCE_THRESHOLD


def predict_letter(landmarks: list[float], language: str = "en") -> dict:
    """
    Predict a letter from 63 landmark floats.

    Args:
        landmarks: List of 63 floats (21 landmarks × 3 coords)
        language: "en" for ASL, "ar" for ArSL

    Returns:
        {"letter": "A", "confidence": 0.95, "all_predictions": [...]}
    """
    # Choose model and encoder based on language
    if language == "ar":
        model = model_manager.arsl_letter_model
        encoder = model_manager.arsl_letter_encoder
    else:
        model = model_manager.asl_letter_model
        encoder = model_manager.asl_letter_encoder

    # Reshape: (63,) → (1, 63)
    input_data = np.array(landmarks).reshape(1, -1)

    # Predict
    predictions = model.predict(input_data, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index])

    # Decode label
    letter = encoder.inverse_transform([predicted_index])[0]

    # Top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [
        {
            "letter": encoder.inverse_transform([i])[0],
            "confidence": float(predictions[i])
        }
        for i in top_indices
    ]

    return {
        "letter": letter,
        "confidence": confidence,
        "above_threshold": confidence >= LETTER_CONFIDENCE_THRESHOLD,
        "top_predictions": top_predictions
    }
```

---

## Step 7: Word Predictor

### `app/models/word_predictor.py`

```python
"""30-frame sequence word prediction using BiLSTM model."""
import numpy as np
from app.models.loader import model_manager
from app.config import WORD_CONFIDENCE_THRESHOLD, SEQUENCE_LENGTH


def predict_word(frame_sequence: list[list[float]], language: str = "en") -> dict:
    """
    Predict a word from a sequence of 30 frames.

    Args:
        frame_sequence: List of 30 frames, each with 63 floats
        language: "en" for English output, "ar" for Arabic output

    Returns:
        {"word_en": "hello", "word_ar": "مرحبا", "confidence": 0.82, ...}
    """
    if len(frame_sequence) != SEQUENCE_LENGTH:
        return {"error": f"Expected {SEQUENCE_LENGTH} frames, got {len(frame_sequence)}"}

    model = model_manager.asl_word_model

    # Reshape: (30, 63) → (1, 30, 63)
    input_data = np.array(frame_sequence).reshape(1, SEQUENCE_LENGTH, -1)

    # Predict
    predictions = model.predict(input_data, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    # Map model index → word_id → vocabulary
    word_id = model_manager.word_classes.get(predicted_index)
    vocab = model_manager.word_vocabulary.get(word_id, {})

    word_en = vocab.get('english', f'unknown_{predicted_index}')
    word_ar = vocab.get('arabic', '')
    category = vocab.get('category', 'unknown')

    # Top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_predictions = []
    for i in top_indices:
        wid = model_manager.word_classes.get(int(i))
        v = model_manager.word_vocabulary.get(wid, {})
        top_predictions.append({
            "word_en": v.get('english', f'unknown_{i}'),
            "word_ar": v.get('arabic', ''),
            "confidence": float(predictions[i])
        })

    return {
        "word": word_ar if language == "ar" else word_en,
        "word_en": word_en,
        "word_ar": word_ar,
        "category": category,
        "confidence": confidence,
        "above_threshold": confidence >= WORD_CONFIDENCE_THRESHOLD,
        "top_predictions": top_predictions
    }
```

---

## Step 8: Mode Detector

### `app/models/mode_detector.py`

```python
"""Detect whether user is signing a letter (still hand) or word (moving hand)."""
import numpy as np
from app.config import MOTION_THRESHOLD_LETTER, MOTION_THRESHOLD_WORD


class ModeDetector:
    """Switches between letter and word mode based on hand motion."""

    def __init__(self):
        self.previous_landmarks = None
        self.current_mode = "letter"  # Start in letter mode
        self.mode_history = []        # Recent mode votes
        self.history_size = 10

    def update(self, landmarks: list[float]) -> str:
        """
        Update with new frame landmarks. Returns current mode.

        Args:
            landmarks: 63 floats for current frame

        Returns:
            "letter" or "word"
        """
        current = np.array(landmarks)

        if self.previous_landmarks is not None:
            # Calculate motion as mean absolute difference
            motion = np.mean(np.abs(current - self.previous_landmarks))

            # Vote for mode
            if motion < MOTION_THRESHOLD_LETTER:
                vote = "letter"
            elif motion > MOTION_THRESHOLD_WORD:
                vote = "word"
            else:
                vote = self.current_mode  # Keep current in dead zone

            self.mode_history.append(vote)
            if len(self.mode_history) > self.history_size:
                self.mode_history.pop(0)

            # Majority vote from recent history
            if self.mode_history:
                letter_count = self.mode_history.count("letter")
                word_count = self.mode_history.count("word")
                self.current_mode = "letter" if letter_count >= word_count else "word"

        self.previous_landmarks = current
        return self.current_mode

    def reset(self):
        """Reset detector state."""
        self.previous_landmarks = None
        self.current_mode = "letter"
        self.mode_history = []
```

---

## Step 9: Letter Decoder

### `app/models/letter_decoder.py`

```python
"""Convert per-frame letter predictions into text with stabilization."""
import time
from collections import deque
from app.config import (
    LETTER_STABLE_WINDOW, LETTER_MAJORITY_RATIO,
    LETTER_COOLDOWN, LETTER_CONFIDENCE_THRESHOLD
)


class LetterDecoder:
    """
    Port of letter_stream_decoder.py — converts frame-by-frame
    predictions into committed text with stabilization.
    """

    def __init__(self):
        self.history = deque(maxlen=LETTER_STABLE_WINDOW)
        self.text = ""
        self.current_word = ""
        self.last_committed = None
        self.last_commit_time = 0.0
        self.control_labels = {"space", "del", "nothing"}

    def update(self, label: str, confidence: float) -> dict:
        """
        Process a new frame prediction.

        Returns:
            {
                "committed": True/False,
                "text": "full text so far",
                "current_word": "current word being built",
                "event": "letter_added" | "space" | "delete" | "none"
            }
        """
        now = time.time()
        event = "none"
        committed = False

        # Skip low confidence
        if confidence < LETTER_CONFIDENCE_THRESHOLD:
            self.history.clear()
            return self._result(committed, event)

        # Skip "nothing"
        if label == "nothing":
            self.history.clear()
            return self._result(committed, event)

        # Add to history
        self.history.append(label)

        # Check if stable (majority agreement)
        if len(self.history) >= LETTER_STABLE_WINDOW:
            from collections import Counter
            counts = Counter(self.history)
            most_common, count = counts.most_common(1)[0]

            ratio = count / len(self.history)
            if ratio >= LETTER_MAJORITY_RATIO:
                # Check cooldown (prevent rapid repeat)
                if most_common == self.last_committed:
                    if now - self.last_commit_time < LETTER_COOLDOWN:
                        return self._result(False, "cooldown")

                # Commit the prediction
                committed = True
                self.last_committed = most_common
                self.last_commit_time = now
                self.history.clear()

                if most_common == "space":
                    self.text += self.current_word + " "
                    self.current_word = ""
                    event = "space"
                elif most_common == "del":
                    if self.current_word:
                        self.current_word = self.current_word[:-1]
                    elif self.text:
                        self.text = self.text.rstrip()
                        # Find last word
                        parts = self.text.rsplit(" ", 1)
                        if len(parts) > 1:
                            self.text = parts[0] + " "
                            self.current_word = parts[1]
                        else:
                            self.current_word = parts[0]
                            self.text = ""
                    event = "delete"
                else:
                    self.current_word += most_common
                    event = "letter_added"

        return self._result(committed, event)

    def _result(self, committed: bool, event: str) -> dict:
        return {
            "committed": committed,
            "text": self.text + self.current_word,
            "full_text": self.text,
            "current_word": self.current_word,
            "event": event
        }

    def clear(self):
        """Reset all state."""
        self.history.clear()
        self.text = ""
        self.current_word = ""
        self.last_committed = None
        self.last_commit_time = 0.0
```

---

## Step 10: Word Decoder

### `app/models/word_decoder.py`

```python
"""Build sentences from word predictions with stability and cooldown."""
import time
from collections import deque
from app.config import WORD_STABILITY_WINDOW, WORD_COOLDOWN, WORD_CONFIDENCE_THRESHOLD


class WordDecoder:
    """Converts word predictions into a sentence with stabilization."""

    def __init__(self):
        self.prediction_history = deque(maxlen=WORD_STABILITY_WINDOW)
        self.sentence_words = []       # Committed words
        self.last_committed_word = None
        self.last_commit_time = 0.0

    def update(self, word_en: str, word_ar: str, confidence: float, language: str = "en") -> dict:
        """
        Process a new word prediction.

        Returns:
            {
                "committed": True/False,
                "sentence": "hello how are you",
                "last_word": "you",
                "event": "word_added" | "none" | "cooldown" | "low_confidence"
            }
        """
        now = time.time()

        # Skip low confidence
        if confidence < WORD_CONFIDENCE_THRESHOLD:
            self.prediction_history.clear()
            return self._result(False, "low_confidence", language)

        word = word_ar if language == "ar" else word_en

        # Add to history
        self.prediction_history.append(word)

        # Check stability (same word predicted N times in a row)
        if len(self.prediction_history) >= WORD_STABILITY_WINDOW:
            if len(set(self.prediction_history)) == 1:
                stable_word = self.prediction_history[0]

                # Check cooldown and no-repeat
                if stable_word == self.last_committed_word:
                    if now - self.last_commit_time < WORD_COOLDOWN:
                        return self._result(False, "cooldown", language)

                # Commit
                self.sentence_words.append(stable_word)
                self.last_committed_word = stable_word
                self.last_commit_time = now
                self.prediction_history.clear()

                return self._result(True, "word_added", language)

        return self._result(False, "none", language)

    def _result(self, committed: bool, event: str, language: str) -> dict:
        separator = " "  # Works for both EN and AR
        sentence = separator.join(self.sentence_words)
        return {
            "committed": committed,
            "sentence": sentence,
            "word_count": len(self.sentence_words),
            "last_word": self.sentence_words[-1] if self.sentence_words else "",
            "event": event
        }

    def clear(self):
        """Reset all state."""
        self.prediction_history.clear()
        self.sentence_words = []
        self.last_committed_word = None
        self.last_commit_time = 0.0
```

---

## Step 11: REST API Routes

### `app/routes/predict.py`

```python
"""REST endpoints for letter and word prediction."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.models.letter_predictor import predict_letter
from app.models.word_predictor import predict_word

router = APIRouter(prefix="/predict", tags=["Prediction"])


class LetterRequest(BaseModel):
    landmarks: list[float] = Field(..., min_length=63, max_length=63)
    language: str = Field(default="en", pattern="^(en|ar)$")


class WordRequest(BaseModel):
    frames: list[list[float]] = Field(..., min_length=30, max_length=30)
    language: str = Field(default="en", pattern="^(en|ar)$")


@router.post("/letter")
async def predict_letter_endpoint(request: LetterRequest):
    """Predict a letter from 63 hand landmark coordinates."""
    try:
        result = predict_letter(request.landmarks, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/word")
async def predict_word_endpoint(request: WordRequest):
    """Predict a word from 30 frames of hand landmarks."""
    try:
        result = predict_word(request.frames, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Step 12: WebSocket Route

### `app/routes/websocket.py`

```python
"""WebSocket endpoint for real-time recognition streaming."""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.models.letter_predictor import predict_letter
from app.models.word_predictor import predict_word
from app.models.mode_detector import ModeDetector
from app.models.letter_decoder import LetterDecoder
from app.models.word_decoder import WordDecoder
from app.config import SEQUENCE_LENGTH

router = APIRouter()


@router.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """
    Real-time recognition over WebSocket.

    Client sends JSON per frame:
        {"landmarks": [63 floats], "language": "en"}

    Server responds:
        {"mode": "letter", "prediction": {...}, "decoder": {...}}
    """
    await websocket.accept()

    # Per-connection state
    mode_detector = ModeDetector()
    letter_decoder = LetterDecoder()
    word_decoder = WordDecoder()
    frame_buffer = []  # Accumulates frames for word prediction

    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            msg = json.loads(data)

            landmarks = msg.get("landmarks", [])
            language = msg.get("language", "en")

            if len(landmarks) != 63:
                await websocket.send_json({"error": "Expected 63 landmarks"})
                continue

            # Detect mode
            mode = mode_detector.update(landmarks)

            response = {"mode": mode}

            if mode == "letter":
                # Clear word buffer when in letter mode
                frame_buffer.clear()

                # Predict letter
                prediction = predict_letter(landmarks, language)
                response["prediction"] = prediction

                # Update letter decoder
                if prediction["above_threshold"]:
                    decoder_result = letter_decoder.update(
                        prediction["letter"],
                        prediction["confidence"]
                    )
                    response["decoder"] = decoder_result

            elif mode == "word":
                # Accumulate frames for word prediction
                frame_buffer.append(landmarks)

                # When we have enough frames, predict
                if len(frame_buffer) >= SEQUENCE_LENGTH:
                    sequence = frame_buffer[-SEQUENCE_LENGTH:]
                    prediction = predict_word(sequence, language)
                    response["prediction"] = prediction

                    # Update word decoder
                    if prediction.get("above_threshold"):
                        decoder_result = word_decoder.update(
                            prediction["word_en"],
                            prediction["word_ar"],
                            prediction["confidence"],
                            language
                        )
                        response["decoder"] = decoder_result

                    # Slide the window (keep last 15 frames for overlap)
                    frame_buffer = frame_buffer[-15:]

                response["frames_buffered"] = len(frame_buffer)
                response["frames_needed"] = SEQUENCE_LENGTH

            # Handle special commands
            if msg.get("command") == "clear":
                letter_decoder.clear()
                word_decoder.clear()
                mode_detector.reset()
                frame_buffer.clear()
                response = {"event": "cleared"}

            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011)
```

---

## Step 13: Health Check Route

### `app/routes/health.py`

```python
"""Health check endpoint."""
from fastapi import APIRouter
from app.models.loader import model_manager

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Check if API and models are loaded."""
    return {
        "status": "healthy",
        "models_loaded": model_manager._loaded,
        "models": {
            "asl_letter": model_manager.asl_letter_model is not None,
            "arsl_letter": model_manager.arsl_letter_model is not None,
            "asl_word": model_manager.asl_word_model is not None,
        },
        "labels": {
            "asl_letters": len(model_manager.asl_letter_encoder.classes_) if model_manager.asl_letter_encoder else 0,
            "arsl_letters": len(model_manager.arsl_letter_encoder.classes_) if model_manager.arsl_letter_encoder else 0,
            "words": len(model_manager.word_vocabulary) if model_manager.word_vocabulary else 0,
        }
    }
```

---

## Step 14: Main App Entry Point

### `app/main.py`

```python
"""ESHARA Sign Language Recognition API — Main entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS, PORT
from app.models.loader import model_manager
from app.routes import predict, websocket, health

# Create FastAPI app
app = FastAPI(
    title="ESHARA Sign Language API",
    description="Bilingual (ASL + ArSL) sign language recognition — Letters & Words",
    version="1.0.0"
)

# CORS (allow web frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(websocket.router)


@app.on_event("startup")
async def startup_event():
    """Load all models when server starts."""
    model_manager.load_all()


@app.get("/")
async def root():
    return {
        "name": "ESHARA Sign Language Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict_letter": "POST /predict/letter",
            "predict_word": "POST /predict/word",
            "websocket": "WS /ws/recognize"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
```

---

## Step 15: Run the Backend

```powershell
cd "m:\Term 10\Grad\Deployment\backend"

# Activate venv
.\venv\Scripts\Activate

# Run server
python -m uvicorn app.main:app --reload --port 8000

# You should see:
# INFO:     Loading models...
# INFO:       ✓ ASL Letter MLP
# INFO:       ✓ ArSL Letter MLP
# INFO:       ✓ ASL Word BiLSTM
# INFO:     All models loaded!
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test It

1. Open browser: `http://localhost:8000/docs` → Interactive Swagger UI
2. Try GET `/health` → Should show all models loaded
3. Try POST `/predict/letter` with 63 dummy floats

---

## Summary

| Step | What You Did | Estimated Time |
|------|-------------|---------------|
| 1 | Virtual environment | 2 min |
| 2 | Install dependencies | 5 min |
| 3-4 | Config + TemporalAttention | 15 min |
| 5 | Model loader | 20 min |
| 6-7 | Letter + Word predictors | 30 min |
| 8 | Mode detector | 15 min |
| 9-10 | Letter + Word decoders | 30 min |
| 11-13 | REST + WebSocket + Health routes | 30 min |
| 14-15 | Main app + run | 10 min |
| **Total** | **Complete backend** | **~3 hours** |

> The backend is the most "familiar" part — it's just Python connecting your existing models to HTTP endpoints.
