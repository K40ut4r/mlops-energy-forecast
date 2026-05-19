"""
Module de feature engineering avance pour la prediction de consommation electrique.
Ajoute des lags, moyennes mobiles, et features temporelles.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/airflow/project"))
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FEATURED_DIR = PROJECT_DIR / "data" / "featured"
FEATURED_DIR.mkdir(parents=True, exist_ok=True)


def add_temporal_features(df):
    """Ajoute des features basees sur le temps (heure, jour, etc.)."""
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "DateTime" in df.columns:
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df = df.set_index("DateTime")
        else:
            raise ValueError("Pas de colonne DateTime trouvee")

    df["hour"] = df.index.hour.astype("float32")
    df["day_of_week"] = df.index.dayofweek.astype("float32")
    df["month"] = df.index.month.astype("float32")
    df["is_weekend"] = (df.index.dayofweek >= 5).astype("float32")
    df["is_night"] = ((df.index.hour >= 22) | (df.index.hour <= 6)).astype("float32")

    return df


def add_lag_features(df, target_col="Global_active_power", lags=[1, 2, 3, 6, 12, 24]):
    """Ajoute des features de lag pour la serie temporelle."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag).astype("float32")
    return df


def add_rolling_features(
    df, target_col="Global_active_power", windows=[3, 6, 12, 24, 168]
):
    """Ajoute des moyennes mobiles et ecarts-types glissants."""
    df = df.copy()
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}h"] = (
            df[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .astype("float32")
        )
        df[f"{target_col}_rolling_std_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).std().astype("float32")
        )
    return df


def add_cyclical_features(df):
    """Encode l'heure et le jour de la semaine de maniere cyclique."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    return df


def build_features(input_path=None, output_path=None):
    """Pipeline complet de feature engineering."""
    if input_path is None:
        input_path = PROCESSED_DIR / "cleaned_data.parquet"
    if output_path is None:
        output_path = FEATURED_DIR / "featured_data.parquet"

    print(f"Chargement des donnees depuis {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   -> {len(df):,} lignes, {len(df.columns)} colonnes")

    print("Ajout des features temporelles...")
    df = add_temporal_features(df)

    print("Ajout des lags...")
    df = add_lag_features(df)

    print("Ajout des moyennes mobiles...")
    df = add_rolling_features(df)

    print("Ajout des features cycliques...")
    df = add_cyclical_features(df)

    initial_len = len(df)
    df = df.dropna()
    print(f"   -> {initial_len - len(df):,} lignes supprimees (NaN des lags)")

    df.to_parquet(output_path, compression="snappy")
    print(f"Donnees sauvegardees : {output_path}")
    print(f"   -> {len(df):,} lignes, {len(df.columns)} features")

    return df


def main():
    build_features()


if __name__ == "__main__":
    main()
