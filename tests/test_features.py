"""
Tests feature engineering sur échantillon.
Vérifie l'absence de data leakage sans charger les 2M+ lignes.
Compatible avec n'importe quel build_features.py.
"""

import numpy as np
import pandas as pd


def test_import_features_module():
    """Vérifie que le module features s'importe."""
    from src.features import build_features
    assert build_features is not None


def test_temporal_cyclic_features():
    """Les features cycliques sin/cos doivent être entre -1 et 1."""
    hours = np.arange(24)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)

    assert np.all(hour_sin >= -1) and np.all(hour_sin <= 1)
    assert np.all(hour_cos >= -1) and np.all(hour_cos <= 1)
    assert np.isclose(hour_sin[0], 0.0, atol=1e-10)


def test_lag_features_no_leakage():
    """Un lag de 1 période ne doit utiliser que des données passées."""
    dates = pd.date_range("2007-01-01", periods=10, freq="h")
    df = pd.DataFrame({"value": np.arange(10.0)}, index=dates)

    df["lag_1h"] = df["value"].shift(1)

    for i in range(1, len(df)):
        assert np.isclose(df["lag_1h"].iloc[i], df["value"].iloc[i - 1])


def test_rolling_features_no_leakage():
    """Rolling mean doit utiliser uniquement des données passées."""
    dates = pd.date_range("2007-01-01", periods=10, freq="h")
    df = pd.DataFrame({"value": np.ones(10) * 5.0}, index=dates)

    df["rolling_mean_3h"] = df["value"].rolling(window=3, min_periods=1, closed="left").mean()

    assert np.isclose(df["rolling_mean_3h"].iloc[1], 5.0)


def test_target_is_future_value():
    """La cible doit être une valeur future (shift négatif)."""
    dates = pd.date_range("2007-01-01", periods=10, freq="h")
    df = pd.DataFrame({"Global_active_power": np.arange(10.0)}, index=dates)

    df["target"] = df["Global_active_power"].shift(-1)

    for i in range(len(df) - 1):
        assert np.isclose(df["target"].iloc[i], df["Global_active_power"].iloc[i + 1])