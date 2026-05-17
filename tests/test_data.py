"""
Tests unitaires pour l'ingestion et le nettoyage des données.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import DataCleaning
from src.data.ingestion import DataIngestion


@pytest.fixture
def mock_raw_data(tmp_path):
    """Crée un fichier de données brutes minimal pour les tests."""
    dates = pd.date_range("2007-01-01 00:00:00", periods=100, freq="min")
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%d/%m/%Y"),
            "Time": dates.strftime("%H:%M:%S"),
            "Global_active_power": np.concatenate([
                np.random.rand(95).astype("float32") * 5,
                [np.nan] * 5,  # 5% missing
            ]),
            "Global_reactive_power": np.random.rand(100).astype("float32"),
            "Voltage": 240 + np.random.randn(100).astype("float32"),
            "Global_intensity": np.random.rand(100).astype("float32"),
            "Sub_metering_1": np.random.rand(100).astype("float32") * 10,
            "Sub_metering_2": np.random.rand(100).astype("float32") * 10,
            "Sub_metering_3": np.random.rand(100).astype("float32") * 10,
        }
    )
    # Simuler les valeurs manquantes UCI (séparateurs vides)
    df.loc[95:97, "Global_active_power"] = np.nan

    file_path = tmp_path / "household_power_consumption.txt"
    df.to_csv(file_path, sep=";", index=False)
    return file_path


class TestDataIngestion:
    """Tests pour l'ingestion des données."""

    def test_load_raw_data(self, mock_raw_data, tmp_path):
        """Vérifie le chargement correct du fichier."""
        ingestion = DataIngestion(
            raw_data_path=mock_raw_data,
            output_dir=tmp_path / "processed",
        )
        df = ingestion.load_raw_data()

        assert len(df) == 100
        assert "Date" in df.columns
        assert "Time" in df.columns
        assert "Global_active_power" in df.columns

    def test_parse_datetime(self, mock_raw_data, tmp_path):
        """Vérifie que Date + Time deviennent un datetime index."""
        ingestion = DataIngestion(
            raw_data_path=mock_raw_data,
            output_dir=tmp_path / "processed",
        )
        df = ingestion.load_raw_data()
        df = ingestion.parse_datetime(df)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert "Date" not in df.columns
        assert "Time" not in df.columns
        assert df.index.is_monotonic_increasing

    def test_numeric_conversion(self, mock_raw_data, tmp_path):
        """Vérifie la conversion en float32."""
        ingestion = DataIngestion(
            raw_data_path=mock_raw_data,
            output_dir=tmp_path / "processed",
        )
        df = ingestion.load_raw_data()
        df = ingestion.parse_datetime(df)
        df = ingestion.convert_numeric_types(df)

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
            assert df[col].dtype == np.float32, f"{col} n'est pas float32"

    def test_file_not_found(self, tmp_path):
        """Vérifie l'erreur si fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            ingestion = DataIngestion(
                raw_data_path=tmp_path / "inexistant.txt",
                output_dir=tmp_path,
            )
            ingestion.load_raw_data()

    def test_run_pipeline(self, mock_raw_data, tmp_path):
        """Test du pipeline complet."""
        ingestion = DataIngestion(
            raw_data_path=mock_raw_data,
            output_dir=tmp_path / "processed",
        )
        df = ingestion.run()

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert (tmp_path / "processed" / "raw_loaded.parquet").exists()


class TestDataCleaning:
    """Tests pour le nettoyage des données."""

    def test_missing_values_detection(self, tmp_path):
        """Vérifie la détection des valeurs manquantes."""
        # Créer un fichier avec des NaN
        dates = pd.date_range("2007-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "Global_active_power": np.concatenate([np.random.rand(45), [np.nan] * 5]),
                "Global_reactive_power": np.random.rand(50),
                "Voltage": 240 + np.random.randn(50),
                "Global_intensity": np.random.rand(50),
                "Sub_metering_1": np.random.rand(50),
                "Sub_metering_2": np.random.rand(50),
                "Sub_metering_3": np.random.rand(50),
                "hour": np.random.randint(0, 24, 50).astype("int8"),
                "day_of_week": np.random.randint(0, 7, 50).astype("int8"),
                "month": np.random.randint(1, 13, 50).astype("int8"),
                "is_weekend": np.random.randint(0, 2, 50).astype("int8"),
            },
            index=dates,
        )
        input_path = tmp_path / "raw_loaded.parquet"
        df.to_parquet(input_path)

        cleaner = DataCleaning(input_path=input_path, output_dir=tmp_path)
        report = cleaner.check_missing_values(df)

        assert len(report) > 0
        assert "Global_active_power" in report.index

    def test_interpolation(self, tmp_path):
        """Vérifie que l'interpolation remplit les trous."""
        dates = pd.date_range("2007-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "Global_active_power": np.concatenate([np.random.rand(45), [np.nan] * 5]),
                "Global_reactive_power": np.random.rand(50),
                "Voltage": 240 + np.random.randn(50),
                "Global_intensity": np.random.rand(50),
                "Sub_metering_1": np.random.rand(50),
                "Sub_metering_2": np.random.rand(50),
                "Sub_metering_3": np.random.rand(50),
                "hour": np.random.randint(0, 24, 50).astype("int8"),
                "day_of_week": np.random.randint(0, 7, 50).astype("int8"),
                "month": np.random.randint(1, 13, 50).astype("int8"),
                "is_weekend": np.random.randint(0, 2, 50).astype("int8"),
            },
            index=dates,
        )
        input_path = tmp_path / "raw_loaded.parquet"
        df.to_parquet(input_path)

        cleaner = DataCleaning(input_path=input_path, output_dir=tmp_path)
        result = cleaner.run(missing_method="interpolate")

        assert result.isnull().sum().sum() == 0, "Il reste des valeurs manquantes !"

    def test_derived_features(self, tmp_path):
        """Vérifie l'ajout des features dérivées."""
        dates = pd.date_range("2007-01-01", periods=50, freq="h")
        df = pd.DataFrame(
            {
                "Global_active_power": np.ones(50, dtype="float32"),  # 1 kW = 1000/60 Wh/min
                "Global_reactive_power": np.random.rand(50),
                "Voltage": 240 + np.random.randn(50),
                "Global_intensity": np.random.rand(50),
                "Sub_metering_1": np.ones(50, dtype="float32") * 10,
                "Sub_metering_2": np.ones(50, dtype="float32") * 10,
                "Sub_metering_3": np.ones(50, dtype="float32") * 10,
                "hour": np.random.randint(0, 24, 50).astype("int8"),
                "day_of_week": np.random.randint(0, 7, 50).astype("int8"),
                "month": np.random.randint(1, 13, 50).astype("int8"),
                "is_weekend": np.random.randint(0, 2, 50).astype("int8"),
            },
            index=dates,
        )
        input_path = tmp_path / "raw_loaded.parquet"
        df.to_parquet(input_path)

        cleaner = DataCleaning(input_path=input_path, output_dir=tmp_path)
        result = cleaner.run()

        assert "Sub_metering_total" in result.columns
        assert "Unmeasured_consumption" in result.columns

        # Vérification : 1 kW * 1000/60 = 16.67 Wh/min
        # Sub_metering_total = 10 + 10 + 10 = 30
        # Unmeasured = 16.67 - 30 = -13.33 (négatif car sub > global, c'est possible)
        assert result["Sub_metering_total"].iloc[0] == 30.0

    def test_no_duplicates(self, tmp_path):
        """Vérifie la suppression des doublons d'index."""
        dates = pd.to_datetime(["2007-01-01 00:00"] * 3 + ["2007-01-01 01:00"] * 2)
        df = pd.DataFrame(
            {
                "Global_active_power": np.random.rand(5),
                "Global_reactive_power": np.random.rand(5),
                "Voltage": 240 + np.random.randn(5),
                "Global_intensity": np.random.rand(5),
                "Sub_metering_1": np.random.rand(5),
                "Sub_metering_2": np.random.rand(5),
                "Sub_metering_3": np.random.rand(5),
                "hour": [0] * 5,
                "day_of_week": [0] * 5,
                "month": [1] * 5,
                "is_weekend": [0] * 5,
            },
            index=dates,
        )
        input_path = tmp_path / "raw_loaded.parquet"
        df.to_parquet(input_path)

        cleaner = DataCleaning(input_path=input_path, output_dir=tmp_path)
        result = cleaner.run()

        assert len(result) == 2  # 2 timestamps uniques
        assert result.index.is_unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])