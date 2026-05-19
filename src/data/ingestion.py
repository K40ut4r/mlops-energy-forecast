"""
Module d'ingestion des données brutes du dataset UCI Household Power Consumption.
"""

import logging
from pathlib import Path

import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Classe responsable du chargement des données brutes."""

    def __init__(
        self, raw_data_path: str | Path, output_dir: str | Path = "data/processed"
    ) -> None:
        """
        Initialise l'ingestion.

        Args:
            raw_data_path: Chemin vers le fichier brut (txt/csv)
            output_dir: Dossier de sortie pour le fichier parquet
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Charge le fichier texte UCI avec le bon séparateur et parsing.

        Returns:
            DataFrame pandas avec les données brutes
        """
        logger.info(f"Chargement des données depuis {self.raw_data_path}")

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {self.raw_data_path}")

        # Le dataset UCI utilise ';' comme séparateur et '?' pour les valeurs manquantes
        df = pd.read_csv(
            self.raw_data_path,
            sep=";",
            low_memory=False,
            na_values=["?", "", " "],
        )

        logger.info(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df

    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine Date + Time en un seul datetime index.

        Args:
            df: DataFrame brut

        Returns:
            DataFrame avec datetime index
        """
        logger.info("Parsing des dates...")

        # Combiner Date et Time
        df["DateTime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )

        # Supprimer les colonnes originales
        df = df.drop(columns=["Date", "Time"])

        # Définir l'index
        df = df.set_index("DateTime").sort_index()

        logger.info(f"Plage temporelle : {df.index.min()} → {df.index.max()}")
        return df

    def convert_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit toutes les colonnes en types numériques appropriés.

        Args:
            df: DataFrame avec types object

        Returns:
            DataFrame avec types float32 (optimisation mémoire)
        """
        logger.info("Conversion des types numériques...")

        numeric_cols = [
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Conversion en float32 pour économiser la RAM (crucial sur 16Go)
        df[numeric_cols] = df[numeric_cols].astype("float32")

        logger.info(
            f"Types convertis. Mémoire utilisée : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )
        return df

    def run(self) -> pd.DataFrame:
        """
        Pipeline complet d'ingestion.

        Returns:
            DataFrame prêt pour le cleaning
        """
        df = self.load_raw_data()
        df = self.parse_datetime(df)
        df = self.convert_numeric_types(df)

        # Sauvegarde intermédiaire
        output_path = self.output_dir / "raw_loaded.parquet"
        df.to_parquet(output_path, compression="snappy")
        logger.info(f"Données brutes sauvegardées : {output_path}")

        return df


def main() -> None:
    """Point d'entrée pour DVC / CLI."""
    ingestion = DataIngestion(
        raw_data_path="data/raw/household_power_consumption.txt",
        output_dir="data/processed",
    )
    ingestion.run()


if __name__ == "__main__":
    main()
