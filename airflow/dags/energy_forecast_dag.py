"""
DAG Airflow pour le pipeline MLOps Energy Forecast.
Orchestre : ingestion → cleaning → features → train → evaluate
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Racine du projet montée dans le conteneur
PROJECT_DIR = "/opt/airflow/project"


def _add_project_to_path():
    """Ajoute le dossier projet au PYTHONPATH pour les imports."""
    project_path = str(Path(PROJECT_DIR))
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def run_ingestion():
    _add_project_to_path()
    from src.data.ingestion import DataIngestion

    ingestion = DataIngestion(
        raw_data_path=os.path.join(PROJECT_DIR, "data", "raw", "household_power_consumption.txt"),
        output_dir=os.path.join(PROJECT_DIR, "data", "processed"),
    )
    ingestion.run()
    return "Ingestion terminée"


def run_cleaning():
    _add_project_to_path()
    from src.data.cleaning import DataCleaning

    cleaner = DataCleaning(
        input_path=os.path.join(PROJECT_DIR, "data", "processed", "raw_loaded.parquet"),
        output_dir=os.path.join(PROJECT_DIR, "data", "processed"),
    )
    cleaner.run()
    return "Cleaning terminé"


def run_features():
    _add_project_to_path()
    from src.features.build_features import build_features

    build_features(
        input_path=os.path.join(PROJECT_DIR, "data", "processed", "cleaned_data.parquet"),
        output_path=os.path.join(PROJECT_DIR, "data", "featured", "featured_data.parquet"),  # ✅ correct arg name + full filename
    )
    return "Features terminées"


def run_training():
    _add_project_to_path()
    from src.models.train import main as train_model
    train_model()
    return "Entraînement terminé"


def run_evaluation():
    _add_project_to_path()
    from src.models.evaluate import main as evaluate_model
    evaluate_model()
    return "Évaluation terminée"


default_args = {
    "owner": "mlops-student",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "energy_forecast_pipeline",
    default_args=default_args,
    description="Pipeline MLOps pour la prédiction de consommation électrique",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 5, 15),
    catchup=False,
    tags=["mlops", "energy", "forecasting"],
) as dag:

    task_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_ingestion,
    )

    task_cleaning = PythonOperator(
        task_id="data_cleaning",
        python_callable=run_cleaning,
    )

    task_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_features,
    )

    task_training = PythonOperator(
        task_id="model_training",
        python_callable=run_training,
    )

    task_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_evaluation,
    )

    task_ingestion >> task_cleaning >> task_features >> task_training >> task_evaluation