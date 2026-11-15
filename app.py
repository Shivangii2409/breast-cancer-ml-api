
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_NAME = "BreastCancerClassifier"

FEATURE_NAMES = ["radius_mean", "texture_mean"]

LABEL_MAP = {
    0: "Benign",
    1: "Malignant"
}

# Read MLflow URL from environment
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:6001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

# Load approved champion model
MODEL_URI = f"models:/{MODEL_NAME}@champion"
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="Breast Cancer Classification API")


class PredictionInput(BaseModel):
    radius_mean: float = Field(gt=0)
    texture_mean: float = Field(gt=0)


@app.get("/")
def home():
    return {
        "message": "Breast Cancer Classification API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/model-info")
def model_info():
    return {
        "model": type(model.named_steps["classifier"]).__name__,
        "features": FEATURE_NAMES,
        "model_name": MODEL_NAME,
        "model_alias": "champion"
    }


@app.post("/predict")
def predict(data: PredictionInput):
    input_data = pd.DataFrame(
        [[data.radius_mean, data.texture_mean]],
        columns=FEATURE_NAMES
    )

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(probabilities[prediction])

    return {
        "prediction": prediction,
        "diagnosis": LABEL_MAP[prediction],
        "confidence": round(confidence, 4)
    }
