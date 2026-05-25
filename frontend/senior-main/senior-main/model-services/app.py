from fastapi import FastAPI
from pydantic import BaseModel

AR_LETTERS = ["ا", "ب", "ت", "ث", "ج"]
EN_LETTERS = ["A", "B", "C", "D", "E"]
AR_WORDS = ["مرحبا", "شكرا", "نعم", "لا"]
EN_WORDS = ["hello", "thanks", "yes", "no"]

app = FastAPI(title="SLR Model Service")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "model-services",
    }


class PredictRequest(BaseModel):
    language: str
    mode: str
    landmarks: list[float]


@app.post("/predict")
def predict(payload: PredictRequest):
    language = payload.language.lower()
    mode = payload.mode.lower()

    if language not in {"ar", "en"} or mode not in {"letters", "words"}:
        return {"message": "Invalid language or mode"}

    if mode == "letters":
        text = AR_LETTERS[0] if language == "ar" else EN_LETTERS[0]
    else:
        text = AR_WORDS[0] if language == "ar" else EN_WORDS[0]

    return {
        "prediction": {
            "text": text,
            "confidence": 0.85,
            "source": "model-service-mock"
        }
    }
