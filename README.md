# ⚡ Smart Energy Forecast - MLOps

[![CI/CD](https://github.com/K40ut4r/mlops-energy-forecast/actions/workflows/ci_cd_pipeline.yml/badge.svg)](https://github.com/K40ut4r/mlops-energy-forecast/actions/workflows/ci_cd_pipeline.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Projet MLOps complet de prédiction de consommation électrique domestique (horizon 1h) avec pipeline CI/CD, monitoring et déploiement Docker.

---

## 📊 Dataset

- **Source** : [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Période** : Décembre 2006 → Novembre 2010
- **Taille** : ~2,075,259 lignes (127 Mo)
- **Lieu** : Sceaux, France
- **Fréquence** : 1 minute
- **Cible** : `Global_active_power` (prédiction à horizon 1h)

---

## 🏗️ Architecture MLOps
Data Ingestion → Cleaning → Feature Engineering → Training → Evaluation → API → Monitoring
↓              ↓              ↓               ↓           ↓         ↓         ↓
DVC/Airflow   Pandas       Pandas/Sklearn   XGBoost    MLflow   FastAPI  Grafana/Evidently


| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Orchestration | Apache Airflow | DAGs ETL + entraînement |
| Data Versioning | DVC | Versionnage du dataset |
| Feature Store | Pandas/Parquet | Feature engineering anti-leakage |
| Modélisation | XGBoost + Scikit-learn | Régression consommation |
| Experiment Tracking | MLflow | Tracking métriques & registry |
| Serving | FastAPI | API REST temps réel |
| UI | Streamlit | Dashboard interactif |
| Monitoring | Evidently + Prometheus | Data drift & performance |
| Conteneurisation | Docker + Docker Compose | Déploiement portable |
| CI/CD | GitHub Actions | Tests + Lint + Build Docker |

---

## 📁 Structure du projet

mlops-energy-forecast/
├── .github/
│   └── workflows/
│       └── ci_cd_pipeline.yml    # CI/CD GitHub Actions
├── airflow/
│   └── dags/
│       └── pipeline_dag.py       # DAG orchestration
├── configs/
│   └── config.yaml               # Configuration centralisée
├── data/
│   ├── raw/                      # Données brutes (DVC tracked)
│   ├── processed/                # Données nettoyées
│   └── featured/                 # Features prêtes pour le ML
├── docker/
│   ├── Dockerfile.api            # Image API FastAPI
│   └── docker-compose.yml        # Stack complète
├── models/                       # Modèles sérialisés (DVC tracked)
├── notebooks/                    # EDA & exploration
├── src/
│   ├── api/
│   │   └── main.py               # API FastAPI
│   ├── data/
│   │   ├── ingestion.py          # Téléchargement UCI
│   │   ├── cleaning.py           # Nettoyage & validation
│   │   └── validation.py         # Great Expectations
│   ├── features/
│   │   └── build_features.py     # Feature engineering (anti-leakage)
│   ├── models/
│   │   ├── train.py              # Entraînement XGBoost + MLflow
│   │   ├── evaluate.py           # Évaluation métriques
│   │   ├── compare_models.py     # Comparaison modèles
│   │   └── local_predict.py      # Inférence locale
│   └── monitoring/
│       ├── drift_detection.py    # Détection data drift
│       └── auto_retrain.py       # Retrain automatique
├── streamlit_app/                # Interface utilisateur
│   ├── app.py
│   ├── pages/
│   │   ├── 01_EDA_Dashboard.py
│   │   ├── 02_Prediction.py
│   │   └── 03_Monitoring.py
│   └── utils/
│       └── api_client.py
├── tests/
│   ├── test_api.py               # Tests API (mock)
│   ├── test_data.py              # Tests data cleaning
│   └── test_features.py          # Tests anti data-leakage
├── requirements.txt              # Dépendances production
├── requirements-test.txt          # Dépendances tests
├── pyproject.toml               # Config pytest, black, isort
└── README.md                     # Ce fichier


---

## 🚀 Quick Start

### Prérequis
- Python 3.11
- Docker & Docker Compose (optionnel)
- ~4 Go RAM libres

### 1. Installation locale

```bash
# Cloner le repo
git clone https://github.com/K40ut4r/mlops-energy-forecast.git
cd mlops-energy-forecast

# Créer l'environnement
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Installer les dépendances de test (optionnel)
pip install -r requirements-test.txt

2. Pipeline DVC (Data + Features + Model)
# Initialiser DVC
dvc init

# Configurer le remote (local pour test, ou S3/GDrive pour prod)
dvc remote add -d myremote /tmp/dvc-storage
# OU
dvc remote add -d myremote gdrive://YOUR_FOLDER_ID

# Reproduire tout le pipeline
dvc repro

3. Airflow (Orchestration)
# Initialiser Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

# Créer un utilisateur admin (utilisez .env en production !)
airflow users create \
  --username admin \
  --password ${AIRFLOW_ADMIN_PASSWORD:-admin} \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

# Lancer le webserver et le scheduler
airflow webserver --port 8080 &
airflow scheduler &

4. MLflow (Tracking)
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

5. FastAPI (Serving)
# Nécessite un modèle entraîné dans models/model.pkl
uvicorn src.api.main:app --reload --port 8000

6. Streamlit (UI)
streamlit run streamlit_app/app.py

7. Docker (Stack complète)
# Lancer toute la stack
docker-compose -f docker/docker-compose.yml up --build

🧪 Tests
# Tous les tests avec couverture
pytest tests/ -v --cov=src --cov-report=term-missing

# Tests spécifiques
pytest tests/test_api.py -v
pytest tests/test_features.py -v
pytest tests/test_data.py -v

🔧 CI/CD Pipeline
Le pipeline GitHub Actions s'exécute à chaque push/PR :
| Job              | Rôle                                 | Déclencheur     |
| ---------------- | ------------------------------------ | --------------- |
| **test**         | Tests unitaires (mock + échantillon) | Push/PR         |
| **build-docker** | Build & test image Docker            | Push sur `main` |

Pourquoi pas d'entraînement automatique dans la CI ?
Le dataset UCI fait 127 Mo (2M+ lignes). L'entraînement XGBoost consomme ~6 Go RAM,
ce qui dépasse les limites des runners GitHub Actions (7 Go max, risque d'OOM kill).
La CI valide la qualité du code et la logique sur des échantillons.
L'entraînement production se fait via dvc repro ou Airflow en local/cloud.


🛡️ Anti Data-Leakage
Le projet implémente des mesures strictes contre le data leakage :
| Mesure                | Implémentation                                         |
| --------------------- | ------------------------------------------------------ |
| **Lags**              | `shift(lag)` avec lag > 0 (données passées uniquement) |
| **Rolling features**  | `rolling(closed="left")` (fenêtre strictement passée)  |
| **Split train/test**  | Split temporel `iloc[:split_idx]` (pas de shuffle)     |
| **Cible**             | `shift(-1)` (valeur future à prédire)                  |
| **Colonnes risquées** | Suppression de `Sub_metering_total` et dérivées        |

👤 Auteur
Kaoutar Mezouahi — Projet de fin de module MLOps
4ème année Ingénierie AI & ML

📄 License
MIT License