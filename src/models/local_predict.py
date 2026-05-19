# src/models/local_predict.py (corrigé)
import pickle
import pandas as pd
from pathlib import Path

def local_predict(features: dict):
    model_path = Path("models/model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Le modèle attend TOUTES les features (46)
    # On crée un DataFrame avec les features fournies, le reste à 0
    expected_features = model.feature_names_in_  # ou liste hardcodée
    
    full_features = {f: features.get(f, 0.0) for f in expected_features}
    df = pd.DataFrame([full_features])
    prediction = model.predict(df)[0]
    
    return {"prediction": float(prediction), "source": "local_dvc_model"}

if __name__ == "__main__":
    # Test avec TOUTES les features requises (ou valeurs par défaut)
    test_features = {
        "Global_active_power": 1.5, "Global_reactive_power": 0.1,
        "Voltage": 240, "Global_intensity": 6.0,
        "Sub_metering_1": 1.0, "Sub_metering_2": 1.0, "Sub_metering_3": 1.0,
        "hour_sin": 0.5, "hour_cos": 0.8, "dow_sin": 0.2, "dow_cos": 0.9,
        "month_sin": 0.3, "month_cos": 0.9, "is_weekend": 0, "day_of_year": 150,
        "lag_1h": 1.5, "lag_2h": 1.4, "lag_6h": 1.3, "lag_24h": 1.2,
        "lag_48h": 1.1, "lag_168h": 1.0,
        "rolling_mean_3h": 1.4, "rolling_std_3h": 0.2,
        "rolling_mean_6h": 1.35, "rolling_std_6h": 0.25,
        "rolling_mean_24h": 1.3, "rolling_std_24h": 0.3,
        "rolling_mean_168h": 1.2, "rolling_std_168h": 0.4,
        "Global_reactive_power_lag_1h": 0.1, "Global_reactive_power_lag_24h": 0.1,
        "Voltage_lag_1h": 240, "Voltage_lag_24h": 240,
        "Global_intensity_lag_1h": 6.0, "Global_intensity_lag_24h": 6.0,
        "Sub_metering_1_lag_1h": 1.0, "Sub_metering_1_lag_24h": 1.0,
        "Sub_metering_2_lag_1h": 1.0, "Sub_metering_2_lag_24h": 1.0,
        "Sub_metering_3_lag_1h": 1.0, "Sub_metering_3_lag_24h": 1.0,
        "ratio_sub1_global": 0.6, "ratio_sub2_global": 0.6, "ratio_sub3_global": 0.6,
        "voltage_intensity_ratio": 40.0,
    }
    result = local_predict(test_features)
    print(f"Prédiction locale : {result['prediction']:.4f} kW")