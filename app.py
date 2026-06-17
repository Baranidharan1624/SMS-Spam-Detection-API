from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np

app = FastAPI()

model = load_model("models/spam_model.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

class SMSRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {
        "message": "SMS Spam Detection API is Running"
    }

@app.post("/predict")
def predict(data: SMSRequest):

    sequence = tokenizer.texts_to_sequences(
        [data.message]
    )

    padded = pad_sequences(
        sequence,
        maxlen=100
    )

    prediction = model.predict(padded)

    result = "Spam" if prediction[0][0] > 0.5 else "Ham"

    return {
        "message": data.message,
        "prediction": result,
        "confidence": float(prediction[0][0])
    }