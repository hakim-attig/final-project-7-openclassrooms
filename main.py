from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd

# =======================
# Chargement du modèle et SHAP explainer
# =======================
model = joblib.load("lightgbm_model.pkl")  # Remplace par ton modèle
explainer = joblib.load("shap_explainer.pkl")
feature_names = joblib.load("feature_names.pkl")  # Liste des noms de features

# =======================
# FastAPI app
# =======================
app = FastAPI(title="API Scoring Crédit")

# =======================
# Models Pydantic
# =======================
class Features(BaseModel):
    features: List[float]

# =======================
# Endpoints
# =======================
@app.get("/status")
def status():
    return {"status": "operational"}

@app.get("/model/info")
def model_info():
    # Exemple d'info, adapte selon ton modèle
    auc_score = 0.85
    optimal_threshold = 0.09
    optimal_cost = 10000
    return {
        "model_type": "lightgbm",
        "auc_score": auc_score,
        "optimal_threshold": optimal_threshold,
        "optimal_cost": optimal_cost
    }

@app.post("/predict")
def predict(features: Features):
    X = np.array([features.features])
    risk_score = model.predict_proba(X)[:,1][0]
    decision = "ACCORD" if risk_score < 0.09 else "REFUS"
    return {
        "risk_score": risk_score,
        "decision": decision,
        "threshold": 0.09
    }

@app.post("/explain")
def explain(features: Features):
    try:
        X = np.array([features.features])
        shap_values = explainer.shap_values(X)
        # On prend les 10 features avec impact absolu le plus élevé
        top_idx = np.argsort(np.abs(shap_values[0]))[::-1][:10]
        top_features = []
        for idx in top_idx:
            f_name = feature_names[idx]
            impact = float(shap_values[0][idx])
            direction = "AUGMENTE LE RISQUE" if impact > 0 else "DIMINUE LE RISQUE"
            top_features.append({"feature": f_name, "impact": impact, "direction": direction})
        return {
            "top_features": top_features,
            "interpretation": "Analyse SHAP générée avec succès"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
