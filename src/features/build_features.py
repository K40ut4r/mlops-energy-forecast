"""
Module de feature engineering pour la prédiction de consommation électrique.

CRITIQUE : Respect strict de la temporalité pour éviter le data leakage.
Aucune feature ne doit utiliser d'information future par rapport au point de prédiction.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Construit les features pour le modèle de prédiction.

    RÈGLE D'OR (anti data-leakage) :
    - Toutes les features doivent être calculées avec des données passées UNIQUEMENT
    - Pas de moyenne mobile centrée, pas de shift négatif
    - Les lags sont positifs (décalage vers le passé)
    """

    def __init__(
        self,
        input_path: str | Path,
        output_dir: str | Path = "data/featured",
        target_col: str = "Global_active_power",
    ) -> None:
        """
        Initialise le feature builder.

        Args:
            input_path: Chemin vers cleaned_data.parquet
            output_dir: Dossier de sortie
            target_col: Colonne cible à prédire
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_col = target_col

    def load_data(self) -> pd.DataFrame:
        """Charge les données nettoyées."""
        logger.info(f"Chargement depuis {self.input_path}")
        return pd.read_parquet(self.input_path)

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features temporelles cycliques (pas de data leakage possible ici).

        Args:
            df: DataFrame avec index datetime

        Returns:
            DataFrame enrichi de features temporelles
        """
        logger.info("Création des features temporelles cycliques...")

        # Heure cyclique (sin/cos) - capture le pattern 24h
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24).astype("float32")
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24).astype("float32")

        # Jour de semaine cyclique
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7).astype("float32")
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7).astype("float32")

        # Mois cyclique (pattern saisonnier)
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12).astype("float32")
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12).astype("float32")

        # Weekend flag
        df["is_weekend"] = (df.index.dayofweek >= 5).astype("int8")

        # Jour de l'année (pattern annuel)
        df["day_of_year"] = df.index.dayofyear.astype("int16")

        logger.info("Features temporelles créées")
        return df

    def create_lag_features(self, df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
        """
        Features de retard (lags) — UNIQUEMENT données passées.

        Args:
            df: DataFrame
            lags: Liste des décalages temporels en heures

        Returns:
            DataFrame avec colonnes lag_Xh
        """
        if lags is None:
            lags = [1, 2, 6, 24, 48, 168]  # 1h, 2h, 6h, 24h, 48h, 1 semaine

        logger.info(f"Création des lags : {lags} heures")

        for lag in lags:
            col_name = f"lag_{lag}h"
            # shift(lag) décale vers le bas = prend la valeur il y a 'lag' périodes
            # C'est correct : pas de data leakage
            df[col_name] = df[self.target_col].shift(lag).astype("float32")

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Features de moyenne mobile sur fenêtre glissante passée.

        ATTENTION : On utilise min_periods=1 pour éviter les NaN au début,
        mais le modèle doit être robuste aux premières lignes.

        Args:
            df: DataFrame
            windows: Fenêtres en heures

        Returns:
            DataFrame avec rolling_mean_Xh, rolling_std_Xh
        """
        if windows is None:
            windows = [3, 6, 24, 168]  # 3h, 6h, 24h, 1 semaine

        logger.info(f"Création des rolling features : {windows} heures")

        for window in windows:
            # rolling(window) utilise les 'window' périodes PRECEDENTES
            # center=False par défaut → pas de data leakage
            df[f"rolling_mean_{window}h"] = (
                df[self.target_col]
                .rolling(window=window, min_periods=1, closed="left")
                .mean()
                .astype("float32")
            )

            df[f"rolling_std_{window}h"] = (
                df[self.target_col]
                .rolling(window=window, min_periods=1, closed="left")
                .std()
                .fillna(0)
                .astype("float32")
            )

        return df

    def create_lag_features_other(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lags sur les autres variables (sub-meterings, voltage, etc.)
        pour capturer les interactions.
        """
        logger.info("Création des lags sur variables exogènes...")

        other_cols = [
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]

        for col in other_cols:
            if col in df.columns:
                df[f"{col}_lag_1h"] = df[col].shift(1).astype("float32")
                df[f"{col}_lag_24h"] = df[col].shift(24).astype("float32")

        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratios entre variables (pas de data leakage)."""
        logger.info("Création des ratios...")

        # Ratio sub-metering / global
        df["ratio_sub1_global"] = (df["Sub_metering_1"] / (df["Global_active_power"] * 1000 / 60 + 1e-6)).astype("float32")
        df["ratio_sub2_global"] = (df["Sub_metering_2"] / (df["Global_active_power"] * 1000 / 60 + 1e-6)).astype("float32")
        df["ratio_sub3_global"] = (df["Sub_metering_3"] / (df["Global_active_power"] * 1000 / 60 + 1e-6)).astype("float32")

        # Tension / Intensité (approximation résistance)
        df["voltage_intensity_ratio"] = (df["Voltage"] / (df["Global_intensity"] + 1e-6)).astype("float32")

        return df

    def resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample les données minute en heure pour réduire la taille
        et aligner sur l'objectif de prédiction horaire.

        Args:
            df: DataFrame minute

        Returns:
            DataFrame horaire
        """
        logger.info("Resampling horaire...")

        # Agrégation horaire
        hourly = df.resample("h").agg({
            "Global_active_power": "mean",
            "Global_reactive_power": "mean",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "sum",  # Wh par heure
            "Sub_metering_2": "sum",
            "Sub_metering_3": "sum",
            "Sub_metering_total": "sum",
            "Unmeasured_consumption": "sum",
        })

        logger.info(f"Resampling effectué : {len(df):,} min → {len(hourly):,} h")
        return hourly

    def remove_leakage_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VÉRIFICATION FINALE : Supprime toute colonne qui pourrait causer du leakage.

        On supprime les colonnes qui sont des transformations directes de la cible
        au même timestamp (ex: Sub_metering_total qui est dérivée de la cible).
        """
        logger.info("Vérification finale anti data-leakage...")

        # Supprimer les colonnes dérivées directement de la cible au même T
        risky_cols = ["Sub_metering_total", "Unmeasured_consumption"]
        for col in risky_cols:
            if col in df.columns:
                logger.warning(f"Suppression de '{col}' (risque de data leakage)")
                df = df.drop(columns=[col])

        return df

    def prepare_target(self, df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
        """
        Prépare la variable cible : prédire la consommation dans 'forecast_horizon' heures.

        Args:
            df: DataFrame
            forecast_horizon: Horizon de prédiction en heures

        Returns:
            DataFrame avec colonne 'target'
        """
        logger.info(f"Préparation de la cible (horizon = {forecast_horizon}h)")

        # La cible est la valeur future → shift(-horizon) décale vers le haut
        # C'est correct : on prédit le futur avec des features passées
        df["target"] = df[self.target_col].shift(-forecast_horizon).astype("float32")

        return df

    def run(self, resample: bool = True, forecast_horizon: int = 1) -> pd.DataFrame:
        """
        Pipeline complet de feature engineering.

        Args:
            resample: Si True, resample en heure
            forecast_horizon: Horizon de prédiction

        Returns:
            DataFrame prêt pour l'entraînement
        """
        df = self.load_data()

        # Optionnel : resampling pour alléger (recommandé sur 16Go RAM)
        if resample:
            df = self.resample_hourly(df)

        # Features
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_lag_features_other(df)
        df = self.create_ratio_features(df)

        # Cible
        df = self.prepare_target(df, forecast_horizon=forecast_horizon)

        # Nettoyage final
        df = self.remove_leakage_risk_features(df)

        # Supprimer les lignes avec NaN (début pour lags, fin pour target)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Suppression des NaN : {initial_len:,} → {len(df):,} lignes")

        # Sauvegarde
        output_path = self.output_dir / "features.parquet"
        df.to_parquet(output_path, compression="snappy")
        logger.info(f"Features sauvegardées : {output_path}")
        logger.info(f"Shape final : {df.shape}")
        logger.info(f"Colonnes : {list(df.columns)}")

        return df


def main() -> None:
    """Point d'entrée pour DVC / CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    builder = FeatureBuilder(input_path="data/processed/cleaned_data.parquet")
    builder.run(resample=True, forecast_horizon=1)


if __name__ == "__main__":
    main()