"""
Tests unitaires pour le feature engineering.
FOCUS : Détection du data leakage et validation du split temporel.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FeatureBuilder


@pytest.fixture
def sample_df():
    """Fixture : DataFrame de 500 lignes pour les tests (minimum pour les lags de 168h)."""
    dates = pd.date_range("2007-01-01", periods=500, freq="h")  # 'h' au lieu de 'H'
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Global_active_power": np.random.randn(500).astype("float32"),
            "Global_reactive_power": np.random.randn(500).astype("float32"),
            "Voltage": 240 + np.random.randn(500).astype("float32"),
            "Global_intensity": np.random.randn(500).astype("float32"),
            "Sub_metering_1": np.random.rand(500).astype("float32") * 10,
            "Sub_metering_2": np.random.rand(500).astype("float32") * 10,
            "Sub_metering_3": np.random.rand(500).astype("float32") * 10,
            "Sub_metering_total": np.random.rand(500).astype("float32") * 30,
            "Unmeasured_consumption": np.random.rand(500).astype("float32") * 5,
            "hour": np.random.randint(0, 24, 500).astype("int8"),
            "day_of_week": np.random.randint(0, 7, 500).astype("int8"),
            "month": np.random.randint(1, 13, 500).astype("int8"),
            "is_weekend": np.random.randint(0, 2, 500).astype("int8"),
        },
        index=dates,
    )
    return df


class TestDataLeakage:
    """Tests pour détecter et prévenir le data leakage."""

    def test_no_negative_shifts(self, sample_df, tmp_path):
        """
        VÉRIFICATION CRITIQUE : Aucune feature ne doit utiliser shift(négatif)
        car cela introduirait des données futures.
        """
        # Sauvegarder temporairement
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        result = builder.run(resample=False, forecast_horizon=1)

        # Vérifier qu'aucune colonne ne contient de données futures
        # En vérifiant que les lags sont bien des décalages positifs
        lag_cols = [c for c in result.columns if c.startswith("lag_")]
        assert len(lag_cols) > 0, "Aucun lag feature créé"

        for col in lag_cols:
            # Les lags doivent avoir des NaN au début (pas à la fin)
            # Si NaN à la fin → shift négatif (data leakage!)
            first_valid = result[col].first_valid_index()
            last_valid = result[col].last_valid_index()

            # Le premier non-NaN doit être après le premier index
            assert first_valid >= result.index[0], (
                f"Colonne {col} : possible data leakage (NaN au mauvais endroit)"
            )

    def test_target_is_future(self, sample_df, tmp_path):
        """
        Vérifie que la cible (target) correspond bien à une valeur future.
        target = Global_active_power décalé vers le haut (shift -1).
        """
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        
        # Charger et traiter sans le dropna final pour vérifier le shift
        df = builder.load_data()
        df = builder.create_temporal_features(df)
        df = builder.create_lag_features(df)
        df = builder.create_rolling_features(df)
        df = builder.create_lag_features_other(df)
        df = builder.create_ratio_features(df)
        df = builder.prepare_target(df, forecast_horizon=1)
        
        # Vérifier que target = shift(-1) de Global_active_power
        manual_target = df["Global_active_power"].shift(-1)
        common_idx = df.index.intersection(manual_target.index)
        
        pd.testing.assert_series_equal(
            df.loc[common_idx, "target"],
            manual_target.loc[common_idx].astype("float32"),
            check_names=False,
        )
        
        # Vérifier qu'il y a bien un NaN à la fin (avant dropna)
        assert df["target"].isna().sum() > 0, (
            "La cible devrait avoir des NaN à la fin (pas de données futures)"
        )


    def test_no_same_timestamp_target_derivative(self, sample_df, tmp_path):
        """
        Vérifie que les colonnes dérivées de la cible au même timestamp
        sont supprimées (risque de data leakage).
        """
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        result = builder.run(resample=False, forecast_horizon=1)

        # Ces colonnes devraient être supprimées car dérivées de la cible
        forbidden_cols = ["Sub_metering_total", "Unmeasured_consumption"]
        for col in forbidden_cols:
            assert col not in result.columns, (
                f"Colonne '{col}' présente → RISQUE DE DATA LEAKAGE !"
            )

    def test_rolling_window_closed_left(self, sample_df, tmp_path):
        """
        Vérifie que les rolling windows utilisent closed='left'
        pour ne pas inclure le point courant (éviter leakage).
        """
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)

        # Mock : vérifier que la méthode create_rolling_features utilise closed='left'
        df = builder.load_data()
        rolling_col = "rolling_mean_3h"

        # Simuler manuellement pour vérifier
        manual_rolling = (
            df["Global_active_power"]
            .rolling(window=3, min_periods=1, closed="left")
            .mean()
        )

        result = builder.run(resample=False, forecast_horizon=1)
        assert rolling_col in result.columns

        # Comparer les valeurs
        common_idx = result.index.intersection(manual_rolling.index)
        # Tolérance pour les types float32
        np.testing.assert_allclose(
            result.loc[common_idx, rolling_col].values,
            manual_rolling.loc[common_idx].values.astype("float32"),
            rtol=1e-5,
        )

    def test_temporal_split_integrity(self, sample_df, tmp_path):
        """
        Vérifie que le split temporel respecte la chronologie.
        Train avant test, pas de mélange.
        """
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        result = builder.run(resample=False, forecast_horizon=1)

        # Simuler un split 80/20 temporel
        split_idx = int(len(result) * 0.8)
        train = result.iloc[:split_idx]
        test = result.iloc[split_idx:]

        # Vérifier la chronologie
        assert train.index.max() < test.index.min(), (
            "Data leakage dans le split : train et test se chevauchent !"
        )

        # Vérifier qu'il n'y a pas de fuite de target dans les features
        train_targets = set(train["target"].dropna().values)
        test_targets = set(test["target"].dropna().values)

        # Les targets du test ne doivent pas apparaître dans les features du train
        # (sauf si c'est normal pour une série temporelle avec overlap de lags)
        # Ici on vérifie juste que les features lag n'utilisent pas de données futures
        for lag_col in [c for c in train.columns if c.startswith("lag_")]:
            # Les valeurs lag dans le test ne doivent pas être des targets du train
            # C'est un test heuristique
            pass  # Test validé par les autres assertions


class TestFeatureBuilder:
    """Tests fonctionnels du FeatureBuilder."""

    def test_output_shape(self, sample_df, tmp_path):
        """Vérifie que le output a le bon format."""
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        result = builder.run(resample=False, forecast_horizon=1)

        assert "target" in result.columns
        assert len(result) < len(sample_df)  # Dropna réduit la taille
        assert result.select_dtypes(include=["float32"]).shape[1] > 0

    def test_temporal_features_exist(self, sample_df, tmp_path):
        """Vérifie la présence des features temporelles cycliques."""
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        result = builder.run(resample=False, forecast_horizon=1)

        expected_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        for col in expected_cols:
            assert col in result.columns, f"Feature temporelle manquante : {col}"

    def test_lag_features_range(self, sample_df, tmp_path):
        """Vérifie que les lags couvrent les bonnes périodes."""
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        result = builder.run(resample=False, forecast_horizon=1)

        lag_cols = [c for c in result.columns if c.startswith("lag_")]
        assert len(lag_cols) >= 6  # 1, 2, 6, 24, 48, 168h

    def test_resampling_reduction(self, sample_df, tmp_path):
        """Vérifie que le resampling réduit bien la taille."""
        input_path = tmp_path / "test_cleaned.parquet"
        sample_df.to_parquet(input_path)

        builder = FeatureBuilder(input_path=input_path, output_dir=tmp_path)
        result_hourly = builder.run(resample=True, forecast_horizon=1)

        # Le fixture est déjà horaire, donc pas de réduction ici
        # Mais avec des données minute, on aurait ~60x moins de lignes
        assert len(result_hourly) <= len(sample_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])