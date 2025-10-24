from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import shap

# --- INITIALISATION DE L'API ---
app = FastAPI(title="API Scoring Crédit", version="1.0")

# --- CHEMIN DU MODÈLE ---
MODEL_PATH = "models/champion_model.pkl"  # ⚠️ Vérifie que ce chemin est correct par rapport à ton main.py

# --- CHARGEMENT DU MODÈLE ---
model = joblib.load(MODEL_PATH)

# --- CRÉATION DE L'EXPLAINER SHAP AU DÉMARRAGE ---
explainer = shap.TreeExplainer(model)

# --- SCHÉMA DES FEATURES ---
class Features(BaseModel):
    features: List[float]

# --- ENDPOINT /status ---
@app.get("/status")
def status():
    return {"status": "operational"}

# --- ENDPOINT /predict ---
@app.post("/predict")
def predict(data: Features):
    try:
        # S'assurer que toutes les features sont des floats
        features = [float(x) for x in data.features]
        X = np.array(features).reshape(1, -1)
        risk_score = model.predict_proba(X)[:,1][0]  # Score entre 0 et 1
        decision = "ACCORD" if risk_score < 0.09 else "REFUS"  # Seuil exemple
        return {"risk_score": float(risk_score), "decision": decision}
    except Exception as e:
        return {"detail": f"Erreur prediction: {str(e)}"}

# --- ENDPOINT /explain ---
@app.post("/explain")
def explain(data: Features):
    try:
        features = [float(x) for x in data.features]
        X = np.array(features).reshape(1, -1)
        shap_values = explainer.shap_values(X)

        # Pour LightGBM binaire : shap_values[1] correspond à la classe positive
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        # Top 10 features les plus importantes
        impact_abs = np.abs(shap_vals[0])
        top_idx = np.argsort(impact_abs)[-10:][::-1]
        top_features = [
            {
                "feature": f"feature_{i}",
                "impact": float(shap_vals[0][i]),
                "direction": "AUGMENTE LE RISQUE" if shap_vals[0][i] > 0 else "DIMINUE LE RISQUE"
            }
            for i in top_idx
        ]

        return {"top_features": top_features, "interpretation": "Explication SHAP générée avec succès."}
    except Exception as e:
        return {"detail": f"Erreur SHAP: {str(e)}"}
