"""
Obesity Level Predictor API
============================
UTS Pembelajaran Mesin - A11.2024.15791

Backend FastAPI untuk memprediksi tingkat obesitas (7 kelas) berdasarkan
kebiasaan makan dan gaya hidup, menggunakan model LightGBM yang sudah
dilatih di notebook Colab.
"""
import logging
import os
import sys
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Logging ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("obesity-api")

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
OE_PATH = os.path.join(MODEL_DIR, "ordinal_encoder.joblib")

ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II", "Obesity_Type_III",
]

X_COLS = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC",
    "MTRANS_Automobile", "MTRANS_Bike", "MTRANS_Motorbike",
    "MTRANS_Public_Transportation", "MTRANS_Walking",
]


def _load_artifact(path: str, human_name: str):
    """Load a joblib artifact with a clear, actionable error if it's missing."""
    if not os.path.exists(path):
        logger.error("File tidak ditemukan: %s", path)
        sys.exit(
            f"\n[STARTUP ERROR] {human_name} tidak ditemukan di '{path}'.\n"
            f"Export file ini dari notebook Colab Anda (lihat TAHAP 1 README) "
            f"dan letakkan di folder 'models/' sebelum menjalankan server.\n"
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # corrupt file, version mismatch, etc.
        logger.error("Gagal memuat %s: %s", path, exc)
        sys.exit(f"\n[STARTUP ERROR] Gagal memuat {human_name} dari '{path}': {exc}\n")


# ── Load model & preprocessing (fail fast & loud if anything is missing) ──
model = _load_artifact(MODEL_PATH, "Model (best_model.joblib)")
scaler = _load_artifact(SCALER_PATH, "Scaler (scaler.joblib)")
oe = _load_artifact(OE_PATH, "Ordinal encoder (ordinal_encoder.joblib)")
logger.info("Model berhasil dimuat: %s", type(model).__name__)

app = FastAPI(
    title="Obesity Level Predictor API",
    description="Memprediksi tingkat obesitas (7 kelas) dari kebiasaan makan & gaya hidup.",
    version="1.0.0",
)

# CORS dibuka untuk semua origin -- aman untuk API publik read-only seperti ini.
# Jika ingin dibatasi, ganti allow_origins dengan daftar domain frontend Anda.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input schema (divalidasi otomatis oleh FastAPI/Pydantic) ──────────────
class PredictInput(BaseModel):
    Gender: Literal["Male", "Female"]
    Age: float = Field(..., ge=1, le=120, description="Usia dalam tahun")
    Height: float = Field(..., ge=0.5, le=2.5, description="Tinggi dalam meter")
    Weight: float = Field(..., ge=10, le=400, description="Berat dalam kilogram")
    family_history_with_overweight: Literal["yes", "no"]
    FAVC: Literal["yes", "no"]
    FCVC: float = Field(..., ge=1, le=3, description="Frekuensi makan sayur (1-3)")
    NCP: float = Field(..., ge=1, le=4, description="Jumlah makan besar/hari (1-4)")
    CAEC: Literal["no", "Sometimes", "Frequently", "Always"]
    SMOKE: Literal["yes", "no"]
    CH2O: float = Field(..., ge=1, le=3, description="Konsumsi air liter/hari (1-3)")
    SCC: Literal["yes", "no"]
    FAF: float = Field(..., ge=0, le=3, description="Frekuensi aktivitas fisik (0-3)")
    TUE: float = Field(..., ge=0, le=2, description="Waktu pakai gadget (0-2)")
    CALC: Literal["no", "Sometimes", "Frequently", "Always"]
    MTRANS: Literal[
        "Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"
    ]


def preprocess(data: PredictInput) -> np.ndarray:
    df = pd.DataFrame([data.model_dump()])

    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    for col in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
        df[col] = df[col].map({"yes": 1, "no": 0})

    df[["CAEC", "CALC"]] = oe.transform(df[["CAEC", "CALC"]])

    df = pd.get_dummies(df, columns=["MTRANS"], prefix="MTRANS")
    for col in X_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[X_COLS]

    mtrans_cols = [c for c in X_COLS if c.startswith("MTRANS_")]
    df[mtrans_cols] = df[mtrans_cols].astype(int)

    return scaler.transform(df)


# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(
            status_code=404,
            content={"detail": "static/index.html tidak ditemukan."},
        )
    return FileResponse(index_path)


@app.post("/predict")
def predict(data: PredictInput):
    try:
        X = preprocess(data)
        pred_idx = int(model.predict(X)[0])
        pred_class = ORDER_TARGET[pred_idx]

        result = {"prediction": pred_class, "probabilities": {}}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            result["probabilities"] = {
                ORDER_TARGET[i]: round(float(p), 4) for i, p in enumerate(proba)
            }
        return result
    except Exception as exc:
        logger.exception("Prediksi gagal")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {exc}")


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}


# Mount static files setelah semua route API didefinisikan
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
