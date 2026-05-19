"""
Tests ingestion et nettoyage (légers, pas besoin du dataset UCI complet).
"""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    convert_types,
    create_derived_columns,
    handle_missing_values,
    validate_data,
)


def test_handle_missing_values_interpolation():
    """L'interpolation linéaire remplit correctement les NaN."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2007-01-01", periods=10, freq="h"),
            "Global_active_power": [1.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    df.set_index("datetime", inplace=True)

    df_clean = handle_missing_values(df)
    assert df_clean.isna().sum().sum() == 0
    assert df_clean["Global_active_power"].iloc[1] > 1.0
    assert df_clean["Global_active_power"].iloc[1] < 4.0


def test_validate_data_detects_negative_power():
    """validate_data doit lever une erreur si puissance négative."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2007-01-01", periods=5, freq="h"),
            "Global_active_power": [1.0, 2.0, -1.0, 4.0, 5.0],
            "Voltage": [240.0] * 5,
            "Global_intensity": [5.0] * 5,
        }
    )
    df.set_index("datetime", inplace=True)

    with pytest.raises(AssertionError):
        validate_data(df)


def test_create_derived_columns():
    """Les colonnes dérivées sont créées correctement."""
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

    df = create_derived_columns(df)
    assert "Sub_metering_total" in df.columns
    assert "Unmeasured_consumption" in df.columns
    assert (df["Sub_metering_total"] == 3.0).all()