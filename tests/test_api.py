"""
Tests API avec modèle factice (mock).
Ne nécessite pas d'entraînement ni de dataset.
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Injecte un modèle factice directement dans le module API."""
    from src.api import main

    dummy = MagicMock()
    dummy.feature_names_in_ = [
        "hour_sin",
        "hour_cos",
        "lag_1h",
        "lag_24h",
        "rolling_mean_24h",
    ]
    dummy.predict = MagicMock(return_value=np.array([1.234]))
    main.model = dummy


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
    assert "prediction" in data
    assert data["unit"] == "kW"


def test_predict_missing_features(client):
    """L'API gère les features manquantes (fill avec 0)."""
    payload = {"features": {"hour_sin": 0.5}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_batch(client):
    from src.api import main

    main.model.predict = MagicMock(return_value=np.array([1.234, 1.234]))

    payload = {
        "items": [
            {"hour_sin": 0.5, "lag_1h": 1.2},
            {"hour_sin": -0.5, "lag_1h": 0.8},
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["count"] == 2


def test_predict_no_model(client):
    """L'API renvoie 503 si le modèle n'est pas chargé."""
    from src.api import main

    original = main.model
    main.model = None
    try:
        payload = {"features": {"hour_sin": 0.5}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
    finally:
        main.model = original
