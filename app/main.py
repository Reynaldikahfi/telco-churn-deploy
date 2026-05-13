# =============================================================
# app/main.py  —  FastAPI Inference Service
# Phase 2: API Layer (One-Hot Encoding version)
# =============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, List
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "model"

try:
    model        = joblib.load(MODEL_DIR / "model.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    prep_info    = joblib.load(MODEL_DIR / "preprocessing_info.pkl")
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)
    THRESHOLD       = meta["threshold"]
    RAW_CAT_COLS    = prep_info["raw_cat_cols"]
    RAW_BINARY_COLS = prep_info["raw_binary_cols"]
    logger.info(f"Model loaded: {meta['model_name']} v{meta['version']} | threshold={THRESHOLD} | fitur={len(feature_cols)}")
except FileNotFoundError as e:
    logger.error(f"Artifact tidak ditemukan: {e}")
    logger.error("Pastikan folder model/ berisi: model.pkl, feature_cols.pkl, preprocessing_info.pkl, model_meta.json")
    raise

app = FastAPI(
    title="Telco Churn Prediction API",
    description="XGBoost model untuk prediksi customer churn",
    version=meta["version"]
)

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None

    @field_validator('gender')
    @classmethod
    def gender_valid(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError("gender harus 'Male' atau 'Female'")
        return v

    @field_validator('SeniorCitizen')
    @classmethod
    def senior_valid(cls, v):
        if v not in [0, 1]:
            raise ValueError("SeniorCitizen harus 0 atau 1")
        return v

class BatchInput(BaseModel):
    customers: List[CustomerInput]

class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_label: str
    threshold_used: float

class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_customers: int
    predicted_churn_count: int

def preprocess(data: dict) -> pd.DataFrame:
    raw = {k: v for k, v in data.items()}

    # Handle TotalCharges kosong
    if raw.get('TotalCharges') is None:
        raw['TotalCharges'] = 29.85
        logger.warning("TotalCharges kosong, diisi median 29.85")

    df = pd.DataFrame([raw])
    df['TotalCharges']   = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(29.85)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['tenure']         = df['tenure'].astype(int)
    df['SeniorCitizen']  = df['SeniorCitizen'].astype(int)

    # One-Hot Encoding — sama persis dengan training
    all_cat_cols = RAW_CAT_COLS + RAW_BINARY_COLS
    df_encoded   = pd.get_dummies(df, columns=all_cat_cols)

    # Align ke kolom training: tambah kolom yang kurang (isi 0), hapus kolom ekstra
    df_aligned = df_encoded.reindex(columns=feature_cols, fill_value=0)

    missing = [c for c in feature_cols if c not in df_encoded.columns]
    if missing:
        logger.warning(f"Kolom tidak ada di input (diisi 0): {missing}")

    return df_aligned

def get_risk_label(prob: float) -> str:
    if prob >= 0.7:   return "High"
    elif prob >= 0.4: return "Medium"
    else:             return "Low"

@app.get("/")
def root():
    return {"api": "Telco Churn Prediction", "version": meta["version"], "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model": meta["model_name"]}

@app.get("/metadata")
def metadata():
    return meta

@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    start = time.time()
    try:
        df_input = preprocess(customer.model_dump())
        prob     = float(model.predict_proba(df_input)[:, 1][0])
        churn    = prob >= THRESHOLD
        logger.info(f"Prediksi: prob={prob:.4f} churn={churn} latency={round((time.time()-start)*1000,2)}ms")
        return PredictionOutput(
            churn_probability=round(prob, 4),
            churn_prediction=churn,
            risk_label=get_risk_label(prob),
            threshold_used=THRESHOLD
        )
    except Exception as e:
        logger.error(f"Error prediksi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    if len(batch.customers) > 500:
        raise HTTPException(status_code=400, detail="Maksimal 500 customer per request")
    predictions = []
    for customer in batch.customers:
        df_input = preprocess(customer.model_dump())
        prob     = float(model.predict_proba(df_input)[:, 1][0])
        churn    = prob >= THRESHOLD
        predictions.append(PredictionOutput(
            churn_probability=round(prob, 4),
            churn_prediction=churn,
            risk_label=get_risk_label(prob),
            threshold_used=THRESHOLD
        ))
    churn_count = sum(1 for p in predictions if p.churn_prediction)
    logger.info(f"Batch {len(predictions)} customers → {churn_count} churn")
    return BatchOutput(predictions=predictions, total_customers=len(predictions), predicted_churn_count=churn_count)