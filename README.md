mlops-energy-forecast/
│
├── .github/
│   └── workflows/
│       └── ci_cd_pipeline.yml          # CI/CD GitHub Actions
│
├── airflow/
│   ├── dags/
│   │   └── energy_forecast_dag.py      # DAG principal
│   └── plugins/
│
├── data/                               # DVC tracked (gitignored)
│   ├── raw/
│   │   └── household_power_consumption.txt
│   ├── processed/
│   │   └── cleaned_data.parquet
│   └── featured/
│       └── features.parquet
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploration initiale
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experimentation.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py                # Chargement CSV
│   │   ├── cleaning.py                 # Gestion missing values
│   │   └── validation.py               # Great Expectations
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py           # Feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                    # Entraînement
│   │   ├── evaluate.py                 # Évaluation
│   │   └── predict.py                  # Prédiction
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                     # FastAPI app
│   │
│   └── monitoring/
│       ├── __init__.py
│       └── drift_detection.py          # Evidently AI
│
├── tests/                              # Tests Pytest
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
│
├── docker/
│   ├── Dockerfile.api                  # Image FastAPI
│   ├── Dockerfile.training             # Image entraînement
│   └── docker-compose.yml
│
├── monitoring/
│   ├── grafana/
│   │   └── dashboards/
│   └── prometheus/
│       └── prometheus.yml
│
├── configs/
│   ├── model_config.yaml               # Hyperparamètres
│   └── airflow_config.yaml
│
├── streamlit_app/                      # Interface Streamlit
│   ├── app.py                          # App principale
│   ├── pages/
│   │   ├── 01_EDA_Dashboard.py
│   │   ├── 02_Prediction.py
│   │   └── 03_Monitoring.py
│   └── utils/
│       └── api_client.py               # Client API FastAPI
│
├── .dvc/                               # DVC metadata
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml             # Pre-commit hooks
├── pyproject.toml                      # Config outils qualité
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── README.md
├── dvc.yaml                            # Pipeline DVC
└── Makefile                            # Commandes utiles
