"""
Module de detection de drift entre les donnees d'entrainement et de test.
Utilise le test de Kolmogorov-Smirnov (KS) et le Population Stability Index (PSI).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/opt/airflow/project"))
FEATURED_DIR = PROJECT_DIR / "data" / "featured"
MONITORING_DIR = PROJECT_DIR / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)


def calculate_psi(expected, actual, buckets=10):
    """
    Calcule le Population Stability Index (PSI).
    PSI < 0.1 : pas de drift
    0.1 <= PSI < 0.25 : drift leger
    PSI >= 0.25 : drift significatif
    """
    def scale_range(input_arr, min_val, max_val):
        return (input_arr - min_val) / (max_val - min_val + 1e-10)

    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_scaled = scale_range(expected, expected.min(), expected.max())
    actual_scaled = scale_range(actual, expected.min(), expected.max())

    expected_counts, _ = np.histogram(expected_scaled, breakpoints)
    actual_counts, _ = np.histogram(actual_scaled, breakpoints)

    expected_percents = np.where(expected_counts == 0, 0.0001, expected_counts) / len(expected)
    actual_percents = np.where(actual_counts == 0, 0.0001, actual_counts) / len(actual)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi)


def detect_drift(train_df, test_df, numeric_cols=None, threshold_ks=0.05, threshold_psi=0.25):
    """Detecte le drift feature par feature."""
    if numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    report = {
        "drift_detected": False,
        "threshold_ks": threshold_ks,
        "threshold_psi": threshold_psi,
        "features": {},
        "summary": {
            "total_features": len(numeric_cols),
            "drifted_features": 0,
            "stable_features": 0,
        },
    }

    for col in numeric_cols:
        if col not in test_df.columns:
            continue

        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()

        if len(train_vals) == 0 or len(test_vals) == 0:
            continue

        ks_stat, p_value = ks_2samp(train_vals, test_vals)
        psi = calculate_psi(train_vals, test_vals)

        # FIX : convertir numpy.bool_ en bool Python natif
        ks_drift = bool(p_value < threshold_ks)
        psi_drift = bool(psi >= threshold_psi)
        is_drifted = bool(ks_drift or psi_drift)

        report["features"][col] = {
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(p_value),
            "psi": float(psi),
            "drift_detected": is_drifted,
            "drift_type": "significant" if psi_drift else ("moderate" if ks_drift else "none"),
            "train_mean": float(train_vals.mean()),
            "test_mean": float(test_vals.mean()),
            "mean_diff_pct": float(abs(train_vals.mean() - test_vals.mean()) / (train_vals.mean() + 1e-10) * 100),
        }

        if is_drifted:
            report["summary"]["drifted_features"] += 1
        else:
            report["summary"]["stable_features"] += 1

    report["drift_detected"] = bool(report["summary"]["drifted_features"] > 0)

    return report


def main():
    """Point d'entree principal."""
    print("Chargement des donnees...")

    featured_files = list(FEATURED_DIR.glob("*.parquet"))
    if len(featured_files) == 0:
        raise FileNotFoundError(f"Aucun fichier parquet trouve dans {FEATURED_DIR}")

    df = pd.read_parquet(featured_files[0])
    print(f"   -> {len(df):,} lignes, {len(df.columns)} colonnes")

    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["Global_active_power"])
    y = df["Global_active_power"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   -> Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("Detection de drift...")
    report = detect_drift(X_train, X_test)

    output_path = MONITORING_DIR / "drift_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Rapport sauvegarde : {output_path}")
    print(f"   -> Features testees : {report['summary']['total_features']}")
    print(f"   -> Features avec drift : {report['summary']['drifted_features']}")
    print(f"   -> Features stables : {report['summary']['stable_features']}")

    if report["drift_detected"]:
        print("DRIFT DETECTE ! Reentrainement recommande.")
        drifted = [k for k, v in report["features"].items() if v["drift_detected"]]
        print(f"   -> Features concernees : {drifted}")
    else:
        print("Aucun drift significatif detecte.")

    return report


if __name__ == "__main__":
    main()