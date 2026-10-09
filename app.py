import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

#configuring API logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease risk based on patient clinical features.",
)

#Instrumenting API to expose /metrics
Instrumentator().instrument(app).expose(app)

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        logger.warning("Model file not found at %s", MODEL_PATH)
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Failed to load model file: %s", exc)
        return None


model = load_model()


class PatientData(BaseModel):
    age: float = Field(..., description="Age in years")
    sex: float = Field(..., description="Sex encoded as 0/1")
    cp: float = Field(..., description="Chest pain type")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol")
    fbs: float = Field(..., description="Fasting blood sugar")
    restecg: float = Field(..., description="Resting ECG result")
    thalach: float = Field(..., description="Maximum heart rate achieved")
    exang: float = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: float = Field(..., description="Slope of peak exercise ST segment")
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy")
    thal: float = Field(..., description="Thalassemia category")


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    risk_label: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData, request: Request):
    #logging the incoming API request
    logger.info(f"Recieved prediction request from {request.client.host}: {data.model_dump()}")


    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found")

    input_df = pd.DataFrame([data.model_dump()])

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction])

        logger.info(f"Prediction generated: {prediction} with confidence {confidence:.4f}")

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "risk_label": "High Risk" if prediction == 1 else "Low Risk",
        }
    except Exception as exc:
        logger.error(f"Inference error: {str(exc)}")
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}") from exc

