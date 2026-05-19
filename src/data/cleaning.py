"""
Module de nettoyage des données : gestion des valeurs manquantes et outliers.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaning:
    """Classe responsable du nettoyage des données."""

    def __init__(
        self, input_path: str | Path, output_dir: str | Path = "data/processed"
    ) -> None:
        """
        Initialise le nettoyage.

        Args:
            input_path: Chemin vers le fichier parquet brut chargé
            output_dir: Dossier de sortie
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Charge les données depuis parquet."""
        logger.info(f"Chargement depuis {self.input_path}")
        return pd.read_parquet(self.input_path)

    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse et rapport des valeurs manquantes.

        Args:
            df: DataFrame à analyser

        Returns:
            DataFrame de statistiques sur les valeurs manquantes
        """
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100

        report = pd.DataFrame(
            {
                "missing_count": missing,
                "missing_pct": missing_pct,
            }
        )
        report = report[report["missing_count"] > 0].sort_values(
            "missing_pct", ascending=False
        )

        logger.info(f"Valeurs manquantes détectées :\n{report}")
        return report

    def handle_missing_values(
        self, df: pd.DataFrame, method: str = "interpolate"
    ) -> pd.DataFrame:
        """
        Gère les valeurs manquantes selon la méthode choisie.

        Pour une série temporelle, l'interpolation linéaire est la méthode
        la plus appropriée car elle respecte la continuité temporelle.

        Args:
            df: DataFrame avec valeurs manquantes
            method: Méthode de remplissage ('interpolate', 'forward_fill', 'drop')

        Returns:
            DataFrame sans valeurs manquantes
        """
        logger.info(f"Gestion des valeurs manquantes par méthode : {method}")

        initial_missing = df.isnull().sum().sum()

        if method == "interpolate":
            # Interpolation linéaire (meilleure pour time series)
            df = df.interpolate(method="time", limit_direction="both")
            # Remplir les bords restants
            df = df.ffill().bfill()

        elif method == "forward_fill":
            df = df.ffill().bfill()

        elif method == "drop":
            df = df.dropna()

        final_missing = df.isnull().sum().sum()
        logger.info(f"Valeurs manquantes : {initial_missing} → {final_missing}")

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les doublons d'index (si existants)."""
        initial_len = len(df)
        df = df[~df.index.duplicated(keep="first")]
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"{removed} doublons supprimés")
        return df

    def handle_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """
        Détecte et traite les outliers (optionnel, à utiliser avec précaution).

        Pour la consommation électrique, on garde les outliers car ce sont
        souvent des événements réels (pic de consommation).

        Args:
            df: DataFrame
            method: Méthode de détection ('iqr', 'zscore', 'none')

        Returns:
            DataFrame traité
        """
        if method == "none":
            logger.info("Pas de traitement des outliers (conservation des pics réels)")
            return df

        logger.info(f"Détection des outliers par méthode : {method}")

        if method == "iqr":
            for col in df.select_dtypes(include=["float32", "float64"]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                logger.info(f"  {col}: {outliers} outliers détectés")

                # Clip plutôt que supprimer (préserve la continuité temporelle)
                df[col] = df[col].clip(lower, upper)

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features dérivées utiles pour l'analyse.

        Args:
            df: DataFrame nettoyé

        Returns:
            DataFrame enrichi
        """
        logger.info("Ajout des features dérivées...")

        # Consommation non mesurée par les sub-meters (différence)
        # Global_active_power est en kW, sub_metering en Wh
        # Conversion : kW * 1000 / 60 = Wh par minute
        df["Sub_metering_total"] = (
            df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]
        )
        df["Unmeasured_consumption"] = (
            df["Global_active_power"] * 1000 / 60 - df["Sub_metering_total"]
        )

        # Features temporelles (utiles pour l'EDA)
        df["hour"] = df.index.hour.astype("int8")
        df["day_of_week"] = df.index.dayofweek.astype("int8")
        df["month"] = df.index.month.astype("int8")
        df["is_weekend"] = (df.index.dayofweek >= 5).astype("int8")

        logger.info("Features dérivées ajoutées")
        return df

    def run(
        self, missing_method: str = "interpolate", outlier_method: str = "none"
    ) -> pd.DataFrame:
        """
        Pipeline complet de nettoyage.

        Args:
            missing_method: Méthode pour les valeurs manquantes
            outlier_method: Méthode pour les outliers

        Returns:
            DataFrame nettoyé et enrichi
        """
        df = self.load_data()

        # Analyse
        self.check_missing_values(df)

        # Nettoyage
        df = self.handle_missing_values(df, method=missing_method)
        df = self.remove_duplicates(df)
        df = self.handle_outliers(df, method=outlier_method)

        # Enrichissement
        df = self.add_derived_features(df)

        # Sauvegarde
        output_path = self.output_dir / "cleaned_data.parquet"
        df.to_parquet(output_path, compression="snappy")
        logger.info(f"Données nettoyées sauvegardées : {output_path}")
        logger.info(f"Shape final : {df.shape}")

        return df


def main() -> None:
    """Point d'entrée pour DVC / CLI."""
    cleaner = DataCleaning(input_path="data/processed/raw_loaded.parquet")
    cleaner.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
