
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
bundle = joblib.load(BASE_DIR / "model.joblib")

model = bundle["model"]
feature_names = bundle["feature_names"]
label_map = bundle["label_map"]

app = FastAPI(title="Breast Cancer Classification API")


class PredictionInput(BaseModel):
    radius_mean: float = Field(gt=0)
    texture_mean: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Breast Cancer Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/model-info")
def model_info():
    return {
        "model": type(model.named_steps["classifier"]).__name__,
        "features": feature_names
    }


@app.post("/predict")
def predict(data: PredictionInput):
    input_data = pd.DataFrame(
        [[data.radius_mean, data.texture_mean]],
        columns=feature_names
    )

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]
    confidence = float(probabilities[prediction])

    return {
        "prediction": prediction,
        "diagnosis": label_map[prediction],
        "confidence": round(confidence, 4)
    }
