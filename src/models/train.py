"""
Module d'entraînement du modèle avec MLflow tracking.
"""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "configs/model_config.yaml") -> dict:
    """Charge la configuration du modèle."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_features(features_path: str | Path = "data/featured/features.parquet") -> pd.DataFrame:
    """Charge les features préparées."""
    logger.info(f"Chargement des features depuis {features_path}")
    return pd.read_parquet(features_path)


def split_temporal(
    df: pd.DataFrame,
    train_end: str = "2009-12-31",
    test_start: str = "2010-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split temporel strict : train avant 2010, test après 2010.
    """
    logger.info(f"Split temporel : train ≤ {train_end}, test ≥ {test_start}")

    train_mask = df.index <= train_end
    test_mask = df.index >= test_start

    train_df = df[train_mask]
    test_df = df[test_mask]

    feature_cols = [c for c in df.columns if c != "target"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    logger.info(f"Train : {len(X_train):,} lignes | Test : {len(X_test):,} lignes")
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les métriques de régression."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> XGBRegressor:
    """Entraîne le modèle XGBoost."""
    logger.info("Entraînement XGBoost...")

    model_params = config["model"]["xgboost"]
    model = XGBRegressor(**model_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    return model


def cross_validate_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    n_splits: int = 5,
) -> dict[str, float]:
    """
    Validation croisée temporelle (TimeSeriesSplit).
    """
    logger.info(f"Cross-validation temporelle ({n_splits} splits)...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**config["model"]["xgboost"])
        model.fit(X_tr, y_tr, verbose=False)

        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val.values, y_pred)
        cv_scores.append(metrics["rmse"])

        logger.info(f"  Fold {fold + 1}: RMSE = {metrics['rmse']:.4f}")

    return {
        "cv_rmse_mean": float(np.mean(cv_scores)),
        "cv_rmse_std": float(np.std(cv_scores)),
    }


def save_model(model: XGBRegressor, output_dir: str | Path = "models") -> Path:
    """Sauvegarde le modèle entraîné."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Modèle sauvegardé : {model_path}")
    return model_path


def main() -> None:
    """Pipeline complet d'entraînement avec MLflow."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("=" * 50)
    logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT")
    logger.info("=" * 50)

    # Créer les dossiers nécessaires
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)

    # Configuration
    config = load_config()

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "energy-forecast-xgboost"
    
    # Récupérer ou créer l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Expérience créée : {experiment_name} (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Expérience existante : {experiment_name} (id={experiment_id})")
    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgboost_baseline"):
        # Log des paramètres
        mlflow.log_params(config["model"]["xgboost"])
        mlflow.log_param("forecast_horizon", "1h")
        mlflow.log_param("split_method", "temporal_2009_2010")

        # Chargement des données
        df = load_features()

        # Split temporel
        X_train, X_test, y_train, y_test = split_temporal(df)

        # Cross-validation
        cv_metrics = cross_validate_temporal(X_train, y_train, config)
        mlflow.log_metrics(cv_metrics)

        # Sauvegarder les métriques d'entraînement pour DVC
        train_metrics = {
            "cv_rmse_mean": cv_metrics["cv_rmse_mean"],
            "cv_rmse_std": cv_metrics["cv_rmse_std"],
        }
        train_metrics_path = Path("metrics/train_metrics.json")
        with open(train_metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=2)
        logger.info(f"Métriques d'entraînement sauvegardées : {train_metrics_path}")

        # Entraînement final
        model = train_model(X_train, y_train, config)

        # Prédictions test
        y_pred = model.predict(X_test)

        # Métriques test
        test_metrics = calculate_metrics(y_test.values, y_pred)
        logger.info(f"Métriques test : {test_metrics}")
        mlflow.log_metrics(test_metrics)

        # Sauvegarde métriques test JSON
        test_metrics_path = Path("metrics/test_metrics.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

        # Feature importance
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance_path = Path("models/feature_importance.csv")
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        # Sauvegarde modèle (pickle)
        model_path = save_model(model)
        mlflow.log_artifact(str(model_path))

        # Log du modèle avec sklearn flavor (évite le bug xgboost)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("=" * 50)
        logger.info("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 50)

        # Affichage résumé
        print("\n" + "=" * 50)
        print("📊 RÉSULTATS")
        print("=" * 50)
        print(f"CV RMSE : {cv_metrics['cv_rmse_mean']:.4f} (+/- {cv_metrics['cv_rmse_std']:.4f})")
        print(f"Test RMSE : {test_metrics['rmse']:.4f}")
        print(f"Test MAE  : {test_metrics['mae']:.4f}")
        print(f"Test MAPE : {test_metrics['mape']:.2f}%")
        print("=" * 50)


if __name__ == "__main__":
    main()