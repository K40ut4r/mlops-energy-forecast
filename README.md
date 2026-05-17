# ⚡ Smart Energy Forecast - MLOps

Projet MLOps de prédiction de consommation électrique domestique.

## 📊 Dataset
- **Source** : UCI Machine Learning Repository
- **Période** : Décembre 2006 → Novembre 2010
- **Taille** : ~2,075,259 lignes (127 Mo)
- **Lieu** : Sceaux, France

## 🏗️ Architecture MLOps
Data Ingestion → Cleaning → Feature Engineering → Training → Evaluation → API → Monitoring
↓              ↓              ↓                ↓           ↓        ↓        ↓
DVC          Pandas         Pandas           XGBoost    MLflow   FastAPI  Grafana
Airflow      Airflow        Airflow          MLflow     Registry Docker  Evidently

## 🚀 Quick Start

### 1. Installation
```bash
# Créer l'environnement
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements-dev.txt

# Setup pre-commit
pre-commit install
2. DVC (Data Version Control)
# Initialiser DVC
dvc init
dvc remote add -d myremote gdrive://YOUR_FOLDER_ID  # ou local
dvc add data/raw/household_power_consumption.txt
git add data/raw/household_power_consumption.txt.dvc .gitignore

3. Airflow
# Initialiser Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
airflow webserver --port 8080 &
airflow scheduler &

4. MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

5. FastAPI
uvicorn src.api.main:app --reload --port 8000

6. Streamlit
streamlit run streamlit_app/app.py

📁 Structure du projet
Voir le schéma complet dans la documentation.
🧪 Tests
pytest tests/ -v --cov=src

🐳 Docker
docker-compose up --build

👤 Auteur
Projet de fin de module MLOps - 4ème année Ingénierie AI & ML


---

#### `requirements.txt`

```txt
# Data & ML
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
xgboost==2.0.3
prophet==1.1.5
tensorflow==2.15.0

# MLOps
mlflow==2.10.0
dvc==3.39.0
apache-airflow==2.8.1
great-expectations==0.18.8

# API & UI
fastapi==0.109.0
uvicorn==0.27.0
streamlit==1.30.0
requests==2.31.0

# Monitoring
evidently==0.4.0
prometheus-client==0.19.0

# Utils
python-dotenv==1.0.0
pyyaml==6.0.1