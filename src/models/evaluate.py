"""
Module d'évaluation du modèle : métriques détaillées et visualisations.
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def load_model(model_path: str | Path = "models/model.pkl"):
    """Charge le modèle sauvegardé."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_test_data(
    features_path: str | Path = "data/featured/features.parquet",
) -> tuple[pd.DataFrame, pd.Series]:
    """Charge les données de test (2010)."""
    df = pd.read_parquet(features_path)
    test_df = df[df.index >= "2010-01-01"]
    feature_cols = [c for c in df.columns if c != "target"]
    return test_df[feature_cols], test_df["target"]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les métriques de régression."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
    }


def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str | Path = "notebooks/figures/predictions_vs_actual.png",
) -> None:
    """Plot prédictions vs réel."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Série temporelle
    sample_idx = slice(0, 500)  # Premieres 500 heures pour visibilité
    axes[0].plot(y_true.index[sample_idx], y_true.values[sample_idx], label="Réel", alpha=0.8)
    axes[0].plot(y_true.index[sample_idx], y_pred[sample_idx], label="Prédit", alpha=0.8)
    axes[0].set_title("Prédictions vs Réel (échantillon 500h)")
    axes[0].set_ylabel("Global Active Power (kW)")
    axes[0].legend()

    # Scatter plot
    axes[1].scatter(y_true.values, y_pred, alpha=0.3, s=1)
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    axes[1].set_xlabel("Réel (kW)")
    axes[1].set_ylabel("Prédit (kW)")
    axes[1].set_title("Prédictions vs Réel (scatter)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot sauvegardé : {output_path}")
    plt.close()


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str | Path = "notebooks/figures/residuals.png",
) -> None:
    """Plot des résidus."""
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution des résidus
    axes[0].hist(residuals, bins=100, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].set_title("Distribution des résidus")
    axes[0].set_xlabel("Résidu (kW)")
    axes[0].axvline(x=0, color="red", linestyle="--")

    # Résidus vs prédit
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=1)
    axes[1].axhline(y=0, color="red", linestyle="--")
    axes[1].set_xlabel("Prédit (kW)")
    axes[1].set_ylabel("Résidu (kW)")
    axes[1].set_title("Résidus vs Prédit")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot sauvegardé : {output_path}")
    plt.close()


def main() -> None:
    """Pipeline d'évaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Évaluation du modèle...")

    # Chargement
    model = load_model()
    X_test, y_test = load_test_data()

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques
    metrics = calculate_metrics(y_test.values, y_pred)
    logger.info(f"Métriques : {metrics}")

    # Sauvegarde des métriques
    metrics_path = Path("metrics/test_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Visualisations
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    # Affichage résumé
    print("\n" + "=" * 50)
    print("📊 MÉTRIQUES DE TEST")
    print("=" * 50)
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"MAPE : {metrics['mape']:.2f}%")
    print(f"R²   : {metrics['r2']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()