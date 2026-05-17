"""
DAG Airflow pour le pipeline MLOps Energy Forecast.
Orchestre : ingestion → cleaning → features → train → evaluate
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "mlops-student",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_ingestion():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/opt/airflow/project")))
    from src.data.ingestion import DataIngestion
    ingestion = DataIngestion(
        raw_data_path="data/raw/household_power_consumption.txt",
        output_dir="data/processed",
    )
    ingestion.run()
    return "Ingestion terminée"


def run_cleaning():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/opt/airflow/project")))
    from src.data.cleaning import DataCleaning
    cleaner = DataCleaning(
        input_path="data/processed/raw_loaded.parquet",
        output_dir="data/processed",
    )
    cleaner.run()
    return "Cleaning terminé"


def run_features():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/opt/airflow/project")))
    from src.features.build_features import FeatureBuilder
    builder = FeatureBuilder(
        input_path="data/processed/cleaned_data.parquet",
        output_dir="data/featured",
    )
    builder.run(resample=True, forecast_horizon=1)
    return "Features terminées"


def run_training():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/opt/airflow/project")))
    from src.models.train import main as train_model
    train_model()
    return "Entraînement terminé"


def run_evaluation():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/opt/airflow/project")))
    from src.models.evaluate import main as evaluate_model
    evaluate_model()
    return "Évaluation terminée"


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