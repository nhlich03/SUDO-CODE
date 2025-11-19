from pathlib import Path
from typing import Optional

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fastapi import FastAPI
from pydantic import BaseModel

MAX_SEQ_LEN = 200

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "sentiment_lstm_imdb.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer_imdb.pkl"

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = load_model(MODEL_PATH)

# FastAPI app
app = FastAPI(
    title="Sentiment LSTM API",
    description="API phân loại cảm xúc review phim (IMDB) bằng LSTM",
    version="1.0"
)

# Schema input
class TextInput(BaseModel):
    text: str


def predict_sentiment(text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post"
    )
    prob = model.predict(padded, verbose=0)[0][0]
    label = 1 if prob >= 0.5 else 0
    sentiment = "positive" if label == 1 else "negative"
    confidence = float(prob if label == 1 else 1 - prob)

    return {
        "text": text,
        "pred_label": int(label),
        "pred_sentiment": sentiment,
        "confidence": confidence
    }


@app.get("/")
def root():
    return {"message": "Sentiment LSTM API is running"}


@app.post("/predict")
def predict(input_data: TextInput):
    result = predict_sentiment(input_data.text)
    return result
