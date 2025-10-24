from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# =======================
# Chemins des fichiers
# =======================
MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/champion_model.pkl"
THRESHOLD_FILE = f"{MODEL_DIR}/champion_threshold.pkl"
FEATURES_FILE = f"{MODEL_DIR}/feature_columns.pkl"
EXPLAINER_FILE = f"{MODEL_DIR}/shap_explainer.pkl"

# =======================
# Chargement du modèle et SHAP explainer
# =======================
model = joblib.load(MODEL_FILE)
threshold = joblib.load(THRESHOLD_FILE)
feature_names = joblib.load(FEATURES_FILE)
explainer = joblib.load(EXPLAINER_FILE)

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
    optimal_cost = 10000
    return {
        "model_type": "lightgbm",
        "auc_score": auc_score,
        "optimal_threshold": threshold,
        "optimal_cost": optimal_cost
    }

@app.post("/predict")
def predict(features: Features):
    if len(features.features) != len(feature_names):
        return JSONResponse(
            status_code=400,
            content={"error": f"Nombre de features incorrect ({len(features.features)} donné, {len(feature_names)} attendu)"}
        )
    X = np.array([features.features])
    risk_score = model.predict_proba(X)[:,1][0]
    decision = "ACCORD" if risk_score < threshold else "REFUS"
    return {
        "risk_score": risk_score,
        "decision": decision,
        "threshold": threshold
    }

@app.post("/explain")
def explain(features: Features):
    try:
        if len(features.features) != len(feature_names):
            return JSONResponse(
                status_code=400,
                content={"error": f"Nombre de features incorrect ({len(features.features)} donné, {len(feature_names)} attendu)"}
            )

        X = np.array([features.features])
        shap_values = explainer.shap_values(X)
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
