⚡ Smart Energy Forecast — Pipeline MLOps
https://www.python.org/
https://airflow.apache.org/
https://fastapi.tiangolo.com/
https://mlflow.org/
https://streamlit.io/
Pipeline MLOps complet pour la prédiction de consommation électrique domestique, basé sur le dataset UCI Household Power Consumption.
🏗️ Architecture
plain
Copy
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ingestion │────▶│   Cleaning  │────▶│   Features  │────▶│  Training   │────▶│ Evaluation  │
│   (Airflow) │     │   (Airflow) │     │  (Airflow)  │     │  (Airflow)  │     │  (Airflow)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                                           │
                                                                                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐                           ┌─────────────┐
│   Streamlit │◀────│   FastAPI   │◀────│    MLflow   │◀────────────────────────│   Model     │
│  (Dashboard)│     │    (API)    │     │  (Tracking) │                           │   (XGBoost) │
└─────────────┘     └─────────────┘     └─────────────┘                           └─────────────┘
Table
Service	Port	Description
Airflow Webserver	8080	Orchestration du pipeline
FastAPI	8000	API de prédiction
MLflow UI	5000	Tracking des expérimentations
Streamlit	8501	Dashboard interactif
Prometheus	9090	Collecte des métriques
Grafana	3000	Visualisation des métriques
📁 Structure du projet
plain
Copy
mlops-energy-forecast/
├── airflow/                  # DAGs et configuration Airflow
│   └── dags/
│       └── energy_forecast_dag.py
├── configs/                  # Fichiers de configuration
├── data/                     # Données (versionnées par DVC)
│   ├── raw/
│   ├── processed/
│   └── featured/
├── docker/                   # Docker Compose et Dockerfiles
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   └── Dockerfile.training
├── metrics/                  # Métriques de performance
├── models/                   # Modèles entraînés
├── monitoring/               # Grafana et Prometheus
├── notebooks/                # Exploration (Jupyter)
├── src/                      # Code source
│   ├── api/                  # FastAPI
│   ├── data/                 # Ingestion et cleaning
│   ├── features/             # Feature engineering
│   ├── models/               # Entraînement et évaluation
│   └── monitoring/           # Drift detection
├── streamlit_app/            # Application Streamlit
├── tests/                    # Tests unitaires
└── .github/workflows/        # CI/CD GitHub Actions
🚀 Démarrage rapide
Prérequis
Docker & Docker Compose
Python 3.11+ (pour le développement local)
Git
1. Cloner le projet
bash
Copy
git clone <url-du-repo>
cd mlops-energy-forecast
2. Lancer tous les services
bash
Copy
cd docker
docker compose up -d
3. Vérifier que tout est up
Table
Service	URL
Airflow	http://localhost:8080 (login: kawme / kawme178)
FastAPI	http://localhost:8000/docs
MLflow	http://localhost:5000
Streamlit	http://localhost:8501
Grafana	http://localhost:3000
4. Lancer le pipeline Airflow
Ouvrir http://localhost:8080
Activer le DAG energy_forecast_pipeline
Cliquer sur ▶️ (Trigger DAG)
Le pipeline s'exécute automatiquement : ingestion → cleaning → features → training → evaluation.
📊 Dashboards
Streamlit (http://localhost:8501)
EDA Dashboard : Exploration des données, statistiques, visualisations
Prediction : Prédiction unitaire et batch via l'API
Monitoring : Métriques du modèle, santé des services, drift detection
Grafana (http://localhost:3000)
Métriques techniques (CPU, mémoire, requêtes API)
Métriques ML (précision, drift, latence des prédictions)
🔮 Utiliser l'API
Health check
bash
Copy
curl http://localhost:8000/health
Prédiction unitaire
bash
Copy
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"Global_active_power": 3.5, "Global_reactive_power": 0.2, "Voltage": 240, "Global_intensity": 15, "Sub_metering_1": 0, "Sub_metering_2": 1, "Sub_metering_3": 17}}'
Prédiction batch
bash
Copy
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"Global_active_power": 3.5, ...}, {"Global_active_power": 0.2, ...}]}'
🧪 Tests
bash
Copy
# Installer les dépendances de test
pip install -r requirements-test.txt

# Lancer les tests
pytest tests/ -v
📈 Résultats du modèle
Table
Métrique	Valeur
RMSE	0.0244
MAE	0.0136
R²	0.9992
Modèle : XGBoost Regressor (100 estimators, max_depth=6)
🔧 Développement local
Sans Docker
bash
Copy
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Lancer les services individuellement
# API
uvicorn src.api.main:app --reload --port 8000

# Streamlit
streamlit run streamlit_app/app.py

# MLflow
mlflow ui --host 0.0.0.0 --port 5000
## 🔄 CI/CD

Le pipeline GitHub Actions s'exécute à chaque push/PR :

| Job | Rôle | Déclencheur |
|-----|------|-------------|
| **Lint** | Black, isort, flake8 | Push/PR |
| **Test** | Tests unitaires (mock + échantillon) | Push/PR |
| **Build** | Build & test Docker image | Push sur `main` |
| **Train** | Entraînement complet sur 2M+ lignes | **Manuel uniquement** |

> **Pourquoi pas d'entraînement automatique ?**
> Le dataset UCI fait 127 Mo (2M+ lignes). L'entraînement XGBoost consomme ~6 Go RAM,
> ce qui dépasse les limites des runners GitHub Actions (7 Go max, risque d'OOM kill).
> La CI valide la qualité du code et la logique sur des échantillons.
> L'entraînement production se fait via `dvc repro` ou Airflow en local/cloud.
📚 Ressources
Dataset UCI
Airflow Documentation
MLflow Documentation
FastAPI Documentation
👤 Auteur
Kaoutar Mezouahi — kmezouahi@gmail.com
Projet réalisé dans le cadre du cursus ISMAGI — CI2 Semestre 2 AI / Machine Learning.