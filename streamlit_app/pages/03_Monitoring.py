import glob
import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Monitoring", page_icon="📈")

st.title("📈 Monitoring & Observabilite")

tab1, tab2, tab3 = st.tabs(["Metriques Modele", "Sante Services", "Drift Detection"])

# ── Tab 1 : Metriques ──
with tab1:
    st.subheader("Metriques de performance")

    for label, path in [("Test", "metrics/test_metrics.json"), ("Train", "metrics/train_metrics.json")]:
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
            st.markdown(f"**{label}**")
            cols = st.columns(len(metrics))
            for idx, (k, v) in enumerate(metrics.items()):
                if isinstance(v, (int, float)):
                    cols[idx].metric(k.upper(), f"{v:.4f}")
                else:
                    cols[idx].write(f"{k}: {v}")
        else:
            st.warning(f"`{path}` introuvable")

    st.divider()
    st.subheader("Logs Airflow recents")
    log_dir = "airflow/logs/dag_id=energy_forecast_pipeline"
    if os.path.exists(log_dir):
        logs = sorted(glob.glob(f"{log_dir}/**/*.log", recursive=True), reverse=True)[:3]
        for log in logs:
            with st.expander(os.path.basename(log)):
                with open(log, "r", encoding="utf-8", errors="ignore") as f:
                    st.text(f.read()[-1500:])
    else:
        st.info("Dossier de logs Airflow non trouve.")

# ── Tab 2 : Sante ──
with tab2:
    st.subheader("Etat des services")

    services = {
        "FastAPI": "http://localhost:8000/health",
        "MLflow": "http://localhost:5000/health",
    }

    for name, url in services.items():
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                st.success(f"{name} : UP")
            else:
                st.warning(f"{name} : HTTP {r.status_code}")
        except Exception:
            st.error(f"{name} : DOWN (`{url}`)")

# ── Tab 3 : Drift (AMELIORE) ──
with tab3:
    st.subheader("Detection de Drift")

    drift_path = "monitoring/drift_report.json"

    if not os.path.exists(drift_path):
        st.info("Aucun rapport de drift trouve. Executez : `python src/monitoring/drift_detection.py`")
        st.stop()

    with open(drift_path) as f:
        report = json.load(f)

    # Resume en haut
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features testees", report["summary"]["total_features"])
    col2.metric("Features stables", report["summary"]["stable_features"])
    col3.metric("Features driftees", report["summary"]["drifted_features"])

    if report["drift_detected"]:
        col4.error("DRIFT DETECTE")
        st.error("⚠️ Drift significatif detecte ! Reentrainement recommande.")
    else:
        col4.success("STABLE")
        st.success("✅ Aucun drift significatif.")

    st.divider()

    # Tableau recapitulatif
    st.subheader("Tableau des features")

    features_data = []
    for feat_name, feat_data in report["features"].items():
        features_data.append({
            "Feature": feat_name,
            "Drift": "🔴 OUI" if feat_data["drift_detected"] else "🟢 Non",
            "Type": feat_data["drift_type"],
            "PSI": round(feat_data["psi"], 4),
            "KS p-value": round(feat_data["ks_p_value"], 4),
            "Train mean": round(feat_data["train_mean"], 4),
            "Test mean": round(feat_data["test_mean"], 4),
            "Diff %": round(feat_data["mean_diff_pct"], 2),
        })

    df_features = pd.DataFrame(features_data)

    # Filtre
    show_only_drift = st.toggle("Afficher uniquement les features avec drift", value=False)
    if show_only_drift:
        df_features = df_features[df_features["Drift"].str.contains("OUI")]

    st.dataframe(df_features, use_container_width=True, hide_index=True)

    st.divider()

    # Graphique PSI
    st.subheader("Score PSI par feature")

    chart_data = df_features.sort_values("PSI", ascending=True).tail(20)  # Top 20

    colors = ["#e74c3c" if "OUI" in d else "#2ecc71" for d in chart_data["Drift"]]

    st.bar_chart(
        data=chart_data.set_index("Feature")["PSI"],
        use_container_width=True,
    )

    st.caption("Seuil PSI = 0.25 (rouge = drift detecte)")

    st.divider()

    # Details JSON
    with st.expander("Voir le rapport JSON brut"):
        st.json(report)