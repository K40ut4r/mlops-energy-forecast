"""
Module d'évaluation du modèle entraîné.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
FEATURED_DIR = Path(PROJECT_DIR) / "data" / "featured"
MODELS_DIR = Path(PROJECT_DIR) / "models"
METRICS_DIR = Path(PROJECT_DIR) / "metrics"


def load_model():
    """Charge le modèle pickle."""
    path = MODELS_DIR / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_test_data(target_col="Global_active_power"):
    """Charge les données featured complètes (on refait un split identique)."""
    from sklearn.model_selection import train_test_split

    path = FEATURED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Données non trouvées : {path}")

    df = pd.read_parquet(path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def evaluate():
    """Évalue le modèle et sauvegarde les métriques."""
    print("📥 Chargement du modèle...")
    model = load_model()

    print("📥 Chargement des données de test...")
    X_test, y_test = load_test_data()

    print("📊 Prédiction et calcul des métriques...")
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_samples": len(y_test),
    }

    # Sauvegarde
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"✅ Évaluation terminée : RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}"
    )
    return metrics


def main():
    return evaluate()


if __name__ == "__main__":
    main()
