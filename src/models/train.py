"""
Module d'entraînement du modèle XGBoost pour la prédiction de consommation électrique.
"""

import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Configuration
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
FEATURED_DIR = Path(PROJECT_DIR) / "data" / "featured"
MODELS_DIR = Path(PROJECT_DIR) / "models"
METRICS_DIR = Path(PROJECT_DIR) / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Charge les données featured."""
    path = FEATURED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Données non trouvées : {path}")
    df = pd.read_parquet(path)
    return df


def split_data(df, target_col="Global_active_power", test_size=0.2, random_state=42):
    """Sépare features et target, puis train/test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    """Entraîne un XGBRegressor."""
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calcule les métriques de performance."""
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return metrics, y_pred


def save_artifacts(model, metrics, X_train):
    """Sauvegarde modèle, métriques et feature importance."""
    # Modèle pickle
    model_path = MODELS_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Métriques JSON
    with open(METRICS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance CSV
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    return model_path


def run_mlflow_experiment(model, metrics, X_train, y_train, X_test, y_test):
    """Logue l'expérience dans MLflow (si disponible)."""
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("energy_forecast")

        with mlflow.start_run():
            # Log params
            mlflow.log_params(
                {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.xgboost.log_model(model, artifact_path="model")

            # Log feature importance
            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame(
                    {
                        "feature": X_train.columns,
                        "importance": model.feature_importances_,
                    }
                )
                importance_path = MODELS_DIR / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))

            run_id = mlflow.active_run().info.run_id
            print(f"✅ MLflow run loggué : {run_id}")
            return run_id

    except Exception as e:
        print(
            f"⚠️ MLflow non disponible ({e}). Modèle sauvegardé localement uniquement."
        )
        return None


def main():
    """Pipeline complet d'entraînement."""
    print("📥 Chargement des données...")
    df = load_data()
    print(f"   → {len(df):,} lignes, {len(df.columns)} colonnes")

    print("🔀 Split train/test...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"   → Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("🚀 Entraînement XGBoost...")
    model = train_model(X_train, y_train)
    print("   → Entraînement terminé")

    print("📊 Évaluation...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    print(
        f"   → RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}"
    )

    print("💾 Sauvegarde des artefacts...")
    model_path = save_artifacts(model, metrics, X_train)
    print(f"   → Modèle: {model_path}")

    print("📝 Log MLflow...")
    _ = mlflow.start_run(...).info.run_id

    print("✅ Entraînement terminé avec succès !")
    return model, metrics


if __name__ == "__main__":
    main()
