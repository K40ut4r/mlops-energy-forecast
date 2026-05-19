# src/models/compare_models.py (optionnel, si tu as 10 min)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Comparaison rapide sur un sous-échantillon
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=50),
    "XGBoost": XGBRegressor(n_estimators=100),
}