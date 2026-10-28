from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Crop Recommendation API",
    description="REST API for crop recommendation based on soil and climate parameters",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_recommendation_model.pkl")
model = None


@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        raise RuntimeError(f"Model not found at {MODEL_PATH}")


class CropInput(BaseModel):
    N: float = Field(..., ge=0, le=200)
    P: float = Field(..., ge=0, le=200)
    K: float = Field(..., ge=0, le=200)
    temperature: float = Field(..., ge=0, le=50)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0, le=500)


class CropOutput(BaseModel):
    crop: str
    confidence: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=CropOutput)
def predict(input_data: CropInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = np.array([[
            input_data.N, input_data.P, input_data.K,
            input_data.temperature, input_data.humidity,
            input_data.ph, input_data.rainfall
        ]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        return CropOutput(crop=str(prediction), confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
