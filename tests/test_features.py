"""
Tests feature engineering sur échantillon.
Vérifie l'absence de data leakage sans charger les 2M+ lignes.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FeatureBuilder


@pytest.fixture
def sample_data(tmp_path):
    """Crée un mini-dataset de 200 lignes pour les tests."""
    dates = pd.date_range("2007-01-01", periods=200, freq="h")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "Global_active_power": np.random.rand(200) * 5,
            "Global_reactive_power": np.random.rand(200) * 2,
            "Voltage": 240 + np.random.rand(200) * 10,
            "Global_intensity": np.random.rand(200) * 20,
            "Sub_metering_1": np.random.rand(200) * 10,
            "Sub_metering_2": np.random.rand(200) * 5,
            "Sub_metering_3": np.random.rand(200) * 15,
            "Sub_metering_total": np.random.rand(200) * 30,
            "Unmeasured_consumption": np.random.rand(200) * 5,
        }
    )
    df.set_index("datetime", inplace=True)
    input_path = tmp_path / "cleaned_data.parquet"
    df.to_parquet(input_path)
    return input_path


def test_lags_use_only_past(sample_data, tmp_path):
    """Lag_1h à l'index i doit être égal à Global_active_power à i-1."""
    builder = FeatureBuilder(
        input_path=sample_data, output_dir=tmp_path / "featured"
    )
    df = builder.load_data()
    df = builder.create_lag_features(df, lags=[1])

    for i in range(2, len(df)):
        assert np.isclose(
            df["lag_1h"].iloc[i],
            df["Global_active_power"].iloc[i - 1],
            rtol=1e-5,
        )


def test_rolling_uses_only_past(sample_data, tmp_path):
    """Rolling mean ne doit pas utiliser de données futures."""
    builder = FeatureBuilder(
        input_path=sample_data, output_dir=tmp_path / "featured"
    )
    df = builder.load_data()
    df = builder.create_rolling_features(df, windows=[3])

    # À l'index 1, rolling_mean_3h = valeur à l'index 0 uniquement
    expected = pd.read_parquet(sample_data)["Global_active_power"].iloc[0]
    actual = df["rolling_mean_3h"].iloc[1]
    assert np.isclose(actual, expected, rtol=1e-5)


def test_risky_columns_removed(sample_data, tmp_path):
    """Sub_metering_total et Unmeasured_consumption doivent être supprimés."""
    builder = FeatureBuilder(
        input_path=sample_data, output_dir=tmp_path / "featured"
    )
    df = builder.load_data()
    df = builder.remove_leakage_risk_features(df)

    assert "Sub_metering_total" not in df.columns
    assert "Unmeasured_consumption" not in df.columns


def test_target_is_future(sample_data, tmp_path):
    """La cible doit être la valeur future (shift -1)."""
    builder = FeatureBuilder(
        input_path=sample_data, output_dir=tmp_path / "featured"
    )
    df = builder.load_data()
    df = builder.prepare_target(df, forecast_horizon=1)

    for i in range(len(df) - 1):
        assert np.isclose(
            df["target"].iloc[i],
            df["Global_active_power"].iloc[i + 1],
            rtol=1e-5,
        )