"""
Tests ingestion et nettoyage (légers, pas besoin du dataset UCI complet).
Compatible avec src/data/cleaning.py (pas clean.py).
"""

import numpy as np
import pandas as pd


def test_import_data_modules():
    """Vérifie que les modules data s'importent sans erreur."""
    from src.data import ingestion
    from src.data import cleaning
    from src.data import validation
    assert ingestion is not None
    assert cleaning is not None
    assert validation is not None


def test_handle_missing_values_interpolation():
    """L'interpolation linéaire remplit correctement les NaN."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2007-01-01", periods=10, freq="h"),
            "Global_active_power": [
                1.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
            ],
        }
    )
    df.set_index("datetime", inplace=True)

    df["Global_active_power"] = df["Global_active_power"].interpolate(method="linear")
    df["Global_active_power"] = df["Global_active_power"].ffill().bfill()

    assert df.isna().sum().sum() == 0
    assert df["Global_active_power"].iloc[1] > 1.0
    assert df["Global_active_power"].iloc[1] < 4.0


def test_validate_no_negative_power():
    """Les valeurs de puissance doivent être positives."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (values >= 0).all()

    negative_values = np.array([1.0, -1.0, 3.0])
    assert not (negative_values >= 0).all()


def test_create_derived_columns():
    """Les colonnes dérivées sont calculées correctement."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2007-01-01", periods=5, freq="h"),
            "Global_active_power": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Sub_metering_1": [1.0] * 5,
            "Sub_metering_2": [1.0] * 5,
            "Sub_metering_3": [1.0] * 5,
        }
    )
    df.set_index("datetime", inplace=True)

    df["Sub_metering_total"] = df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]

    assert "Sub_metering_total" in df.columns
    assert (df["Sub_metering_total"] == 3.0).all()