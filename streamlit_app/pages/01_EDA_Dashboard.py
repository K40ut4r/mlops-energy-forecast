import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="EDA Dashboard", page_icon="📊")

st.title("📊 Analyse Exploratoire des Données")

# Cherche les données
DATA_PATH = os.environ.get("DATA_PATH", "data/processed/cleaned_data.parquet")

if not os.path.exists(DATA_PATH):
    st.warning("Aucune donnée trouvée. Lancez d'abord le pipeline Airflow.")
    st.info(f"Chemin cherché : `{DATA_PATH}`")
    st.stop()

df = pd.read_parquet(DATA_PATH)
st.success(f"Données chargées : {len(df):,} lignes, {len(df.columns)} colonnes")

# Aperçu
with st.expander("Voir les 20 premières lignes"):
    st.dataframe(df.head(20))

# Stats
st.subheader("Statistiques descriptives")
st.write(df.describe())

# Graphique simple
st.subheader("Consommation globale (premiers 1000 points)")
if "Global_active_power" in df.columns:
    st.line_chart(df["Global_active_power"].head(1000))
else:
    st.info("Colonne `Global_active_power` non trouvée.")
