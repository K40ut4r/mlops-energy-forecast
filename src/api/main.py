"""
API FastAPI pour servir le modèle de prédiction de consommation électrique.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Smart Energy Forecast API",
    description="API de prédiction de consommation électrique domestique",
    version="1.0.0",
)


# Chargement du modèle au démarrage
MODEL_PATH = Path("models/model.pkl")
model = None

if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)


class PredictionRequest(BaseModel):
    """Requête de prédiction."""
    features: dict  # Dict des features {nom_feature: valeur}


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    prediction: float
    unit: str = "kW"
    horizon: str = "1h"


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
        
        # Vérifier que toutes les features attendues sont présentes
        expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
        
        # Réordonner selon le modèle
        df = df.reindex(columns=expected_features, fill_value=0)
        
        # Prédiction
        prediction = model.predict(df.values)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            unit="kW",
            horizon="1h",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    """Prédiction en batch."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    predictions = []
    for req in requests:
        df = pd.DataFrame([req.features])
        pred = model.predict(df.values)[0]
        predictions.append(float(pred))
    
    return {"predictions": predictions, "unit": "kW", "horizon": "1h"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)