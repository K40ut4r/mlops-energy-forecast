"""
Script de reentrainement automatique declenche quand le drift depasse un seuil.
A integrer dans un DAG Airflow ou executer via cron.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import requests

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/airflow/project"))
MONITORING_DIR = PROJECT_DIR / "monitoring"
DRIFT_REPORT = MONITORING_DIR / "drift_report.json"

# Seuils
DRIFT_THRESHOLD = 0.25  # PSI max acceptable
MAX_DRIFTED_FEATURES = 3  # Nombre max de features driftees avant reentrainement

# URLs des services
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080")
API_URL = os.environ.get("API_URL", "http://api:8000")


def load_drift_report():
    """Charge le rapport de drift."""
    if not DRIFT_REPORT.exists():
        print("Rapport de drift non trouve ")
        return None
    with open(DRIFT_REPORT, "r") as f:
        return json.load(f)


def check_drift(report):
    """Verifie si le drift necessite un reentrainement."""
    if report is None:
        return False, "Pas de rapport de drift disponible"

    drifted = [k for k, v in report["features"].items() if v["drift_detected"]]

    # Critere 1 : nombre de features driftees
    if len(drifted) > MAX_DRIFTED_FEATURES:
        return True, len(drifted) + " features avec drift " + MAX_DRIFTED_FEATURES

    # Critere 2 : PSI moyen eleve
    avg_psi = sum(v["psi"] for v in report["features"].values()) / len(
        report["features"]
    )
    if avg_psi > 0.1:
        return True, f"PSI moyen eleve : {avg_psi:.4f}"

    # Critere 3 : feature critique driftee
    critical_features = ["Global_active_power", "Voltage", "Global_intensity"]
    critical_drifted = [f for f in drifted if any(c in f for c in critical_features)]
    if len(critical_drifted) > 0:
        return True, f"Features critiques driftees : {critical_drifted}"

    return False, f"Drift acceptable ({len(drifted)} features, PSI moyen={avg_psi:.4f})"


def trigger_airflow_dag(dag_id="energy_forecast_pipeline"):
    """Declenche le DAG Airflow via l'API."""
    try:
        url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
        resp = requests.post(
            url,
            json={"conf": {}},
            auth=("kawme", "kawme178"),
            timeout=10,
        )
        resp.raise_for_status()
        run_id = resp.json().get("dag_run_id", "unknown")
        print(f"DAG declenche : {run_id}")
        return True
    except Exception as e:
        print(f"Erreur lors du declenchement du DAG : {e}")
        return False


def retrain_locally():
    """Reentraine le modele localement (fallback si Airflow indisponible)."""
    print("Reentrainement local...")
    scripts = [
        "src/features/build_features.py",
        "src/models/train.py",
        "src/models/evaluate.py",
    ]
    for script in scripts:
        path = PROJECT_DIR / script
        if path.exists():
            print(f"  Execution de {script}...")
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("  ERREUR : " + result.stderr)
                return False
            print("  OK")
        else:
            print("  Script non trouve : " + path)
    return True


def reload_api_model():
    """Notifie l'API de recharger le modele (si endpoint disponible)."""
    try:
        resp = requests.post(API_URL + "/reload", timeout=5)
        if resp.status_code == 200:
            print("Modele API recharge")
        else:
            print("Rechargement API non disponible (HTTP )" + resp.status_code)
    except Exception:
        print("API non accessible pour rechargement")


def main():
    """Point d'entree principal."""
    print("=" * 50)
    print("VERIFICATION DU DRIFT ET REENTRAINEMENT")
    print("=" * 50)

    # 1. Charger le rapport
    print("[1/4] Chargement du rapport de drift...")
    report = load_drift_report()

    # 2. Verifier si reentrainement necessaire
    print("[2/4] Analyse du drift...")
    needs_retrain, reason = check_drift(report)
    print("Resultat :" + reason)

    if not needs_retrain:
        print("✅ Aucun reentrainement necessaire.")
        return 0

    print("⚠️REENTRAINEMENT NECESSAIRE :" + reason)

    # 3. Declencher le reentrainement
    print("[3/4] Declenchement du reentrainement...")

    # Essayer Airflow d'abord
    if trigger_airflow_dag():
        print("✅ Pipeline Airflow declenche avec succes")
    else:
        print("⚠️  Airflow indisponible, reentrainement local...")
        if retrain_locally():
            print("✅ Reentrainement local termine")
            reload_api_model()
        else:
            print("❌ Echec du reentrainement local")
            return 1

    # 4. Mettre a jour le rapport
    print("[4/4] Regeneration du rapport de drift...")
    drift_script = PROJECT_DIR / "src" / "monitoring" / "drift_detection.py"
    if drift_script.exists():
        subprocess.run([sys.executable, str(drift_script)], capture_output=True)
        print("✅ Rapport de drift mis a jour")

    print("" + "=" * 50)
    print("REENTRAINEMENT TERMINE")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
