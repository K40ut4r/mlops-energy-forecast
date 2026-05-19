"""
Tests API avec modèle factice (mock).
Ne nécessite pas d'entraînement ni de dataset.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Mock du modèle avant import
from src.api import main


class DummyModel:
    """Modèle factice pour les tests."""

    def __init__(self):
        self.feature_names_in_ = [
            "hour_sin",
            "hour_cos",
            "lag_1h",
            "lag_24h",
            "rolling_mean_24h",
        ]

    def predict(self, X):
        return np.array([1.234] * len(X))


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Crée un modèle factice avant les tests."""
    artifact = {
        "model": DummyModel(),
        "feature_names": [
            "hour_sin",
            "hour_cos",
            "lag_1h",
            "lag_24h",
            "rolling_mean_24h",
        ],
    }
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    # Recharger dans l'API
    main.load_model()


@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_single(client):
    payload = {
        "features": {
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "lag_1h": 1.2,
            "lag_24h": 0.9,
            "rolling_mean_24h": 1.1,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1.234
    assert data["unit"] == "kW"


def test_predict_missing_features(client):
    """Test que l'API gère les features manquantes (fill avec 0)."""
    payload = {"features": {"hour_sin": 0.5}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_batch(client):
    payload = {
        "items": [
            {"hour_sin": 0.5, "lag_1h": 1.2},
            {"hour_sin": -0.5, "lag_1h": 0.8},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["count"] == 2


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert len(data["feature_names"]) == 5