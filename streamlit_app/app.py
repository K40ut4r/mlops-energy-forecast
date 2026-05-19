import streamlit as st

st.set_page_config(page_title="Energy Forecast", page_icon="⚡")

st.title("⚡ Smart Energy Forecast")
st.markdown(
    """
Bienvenue dans l'application MLOps de prédiction de consommation électrique.

**Architecture :**
- **Airflow** : Orchestration du pipeline
- **FastAPI** : API de prédiction (`http://localhost:8000`)
- **MLflow** : Tracking des expérimentations (`http://localhost:5000`)
- **Streamlit** : Cette interface

**Navigation** (menu à gauche) :
- **EDA Dashboard** : Explorer les données
- **Prediction** : Faire des prédictions via l'API
- **Monitoring** : Voir les métriques et la santé des services
"""
)

st.info(
    "Assurez-vous que l'API FastAPI tourne sur `http://localhost:8000` avant de faire des prédictions."
)
