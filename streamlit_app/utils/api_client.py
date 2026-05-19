import os
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def health_check():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None


def predict(features: dict):
    r = requests.post(
        f"{API_URL}/predict",
        json={"features": features},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def predict_batch(items: list):
    r = requests.post(
        f"{API_URL}/predict_batch",
        json={"items": items},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()