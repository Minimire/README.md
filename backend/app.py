import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("crop_api")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_recommendation_model.pkl")

# Must match the EXACT column order/names used during training
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"Model {type(model).__name__} has no predict_proba — classifier required"
        )

    ml["model"] = model
    logger.info(
        "Model loaded: %s | classes=%s",
        type(model).__name__,
        getattr(model, "classes_", "?"),
    )
    try:
        yield
    finally:
        ml.clear()


app = FastAPI(
    title="Crop Recommendation API",
    description="REST API for crop recommendation based on soil and climate parameters",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CropInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    N: float = Field(..., ge=0, le=200)
    P: float = Field(..., ge=0, le=200)
    K: float = Field(..., ge=0, le=200)
    temperature: float = Field(..., ge=-10, le=60)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0, le=500)


class CropOutput(BaseModel):
    crop: str
    confidence: float


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health_check():
    if ml.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=CropOutput)
def predict(input_data: CropInput):
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # DataFrame preserves feature names → matches training, avoids warnings
    features = pd.DataFrame(
        [[
            input_data.N, input_data.P, input_data.K,
            input_data.temperature, input_data.humidity,
            input_data.ph, input_data.rainfall,
        ]],
        columns=FEATURE_ORDER,
    )

    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        return CropOutput(crop=str(prediction), confidence=float(np.max(probabilities)))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")