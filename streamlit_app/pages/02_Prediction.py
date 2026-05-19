import os

import pandas as pd
import requests
import streamlit as st
from utils.api_client import health_check, predict

st.set_page_config(page_title="Prediction", page_icon="🔮")

st.title("🔮 Prédiction de Consommation")

# Vérifie API
health = health_check()
if health is None:
    st.error("Impossible de contacter l'API sur `http://localhost:8000`.")
    st.info("Démarrez l'API : `docker compose -f docker/docker-compose.yml up -d api`")
    st.stop()

st.success(f"API connectée (model_loaded={health.get('model_loaded', False)})")

# Valeurs par défaut
st.subheader("Saisie des features")
col1, col2 = st.columns(2)
with col1:
    gap = st.number_input("Global Active Power (kW)", value=1.09, step=0.1)
    grp = st.number_input("Global Reactive Power (kVAR)", value=0.16, step=0.1)
    voltage = st.number_input("Voltage (V)", value=240.38, step=1.0)
    intensity = st.number_input("Global Intensity (A)", value=4.63, step=0.1)
with col2:
    sm1 = st.number_input("Sub Metering 1 (Wh)", value=1.12, step=0.1)
    sm2 = st.number_input("Sub Metering 2 (Wh)", value=1.30, step=0.1)
    sm3 = st.number_input("Sub Metering 3 (Wh)", value=6.46, step=0.1)

features = {
    "Global_active_power": gap,
    "Global_reactive_power": grp,
    "Voltage": voltage,
    "Global_intensity": intensity,
    "Sub_metering_1": sm1,
    "Sub_metering_2": sm2,
    "Sub_metering_3": sm3,
}

if st.button("Prédire", type="primary"):
    with st.spinner("Appel API en cours..."):
        try:
            result = predict(features)
            st.success(f"Prédiction : **{result['prediction']:.4f} kW**")
            st.write(
                f"Unité : {result.get('unit', 'kW')} | Horizon : {result.get('horizon', '1h')}"
            )
        except Exception as e:
            st.error(f"Erreur : {e}")

# Batch
st.divider()
st.subheader("Prédiction en batch (CSV)")
uploaded = st.file_uploader("Uploader un CSV", type=["csv"])
if uploaded:
    df_up = pd.read_csv(uploaded)
    st.write(f"{len(df_up)} lignes chargées")
    st.dataframe(df_up.head())

    required = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    missing = [c for c in required if c not in df_up.columns]
    if missing:
        st.error(f"Colonnes manquantes : {missing}")
    elif st.button("Prédire le batch"):
        with st.spinner("Prédiction en cours..."):
            try:
                items = df_up[required].to_dict(orient="records")
                r = requests.post(
                    f"{os.environ.get('API_URL', 'http://localhost:8000')}/predict_batch",
                    json={"items": items},
                    timeout=30,
                )
                r.raise_for_status()
                preds = r.json()["predictions"]
                df_up["prediction_kW"] = preds
                st.success(f"{len(preds)} prédictions effectuées")
                st.dataframe(df_up)
                csv = df_up.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger CSV", csv, "predictions.csv", "text/csv"
                )
            except Exception as e:
                st.error(f"Erreur batch : {e}")
