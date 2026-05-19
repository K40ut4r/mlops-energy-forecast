"""
API FastAPI pour servir le modèle de prédiction de consommation électrique.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Smart Energy Forecast API",
    description="API de prédiction de consommation électrique domestique",
    version="1.0.0",
)

# ─────────────────────────────────────────
# Chargement du modèle au démarrage
# ─────────────────────────────────────────
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pkl"))
model = None

if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Modèle chargé depuis {MODEL_PATH}")
else:
    print(
        f"⚠️ Modèle non trouvé à {MODEL_PATH}. L'API tourne mais /predict renverra une erreur."
    )


# ─────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Requête de prédiction."""

    features: Dict[
        str, float
    ]  # ex: {"Global_active_power": 2.5, "Voltage": 240.0, ...}


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""

    prediction: float
    unit: str = "kW"
    horizon: str = "1h"


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction en batch."""

    items: List[Dict[str, float]]


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "api_version": "1.0.0",
    }


@app.get("/health")
def health():
    """Health check pour monitoring."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédire la consommation électrique."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # Convertir en DataFrame (1 ligne)
        df = pd.DataFrame([request.features])

        # Réordonner selon les features attendues par le modèle
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            df = df.reindex(columns=expected_features, fill_value=0)

        # Prédiction
        prediction = float(model.predict(df.values)[0])

        return PredictionResponse(
            prediction=prediction,
            unit="kW",
            horizon="1h",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict_batch")
def predict_batch(request: BatchPredictionRequest):
    """Prédiction en batch."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        df = pd.DataFrame(request.items)

        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            df = df.reindex(columns=expected_features, fill_value=0)

        predictions = model.predict(df.values).tolist()
        return {
            "predictions": [float(p) for p in predictions],
            "unit": "kW",
            "horizon": "1h",
            "count": len(predictions),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur de prédiction batch: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
