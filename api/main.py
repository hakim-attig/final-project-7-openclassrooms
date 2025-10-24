from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import shap
import os

# --- CHEMINS DYNAMIQUES ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dossier projet_scoring_credit
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "champion_model.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "champion_threshold.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# --- INITIALISATION DE L'API ---
app = FastAPI(title="API Scoring Crédit", version="1.0")

# --- CHARGEMENT DU MODÈLE ---
try:
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    model_loaded = True
    print(f"✓ Modèle chargé: {model}, Seuil optimal: {threshold}, Nombre de features: {len(feature_columns)}")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")
    model_loaded = False
    threshold = None
    feature_columns = []

# --- CRÉATION DE L'EXPLAINER SHAP ---
try:
    if model_loaded:
        explainer = shap.TreeExplainer(model)
        print("✓ Explainer SHAP créé avec succès")
except Exception as e:
    print(f"Erreur création SHAP Explainer: {e}")
    explainer = None

# --- SCHÉMA DES FEATURES ---
class Features(BaseModel):
    features: List[float]

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {
        "api": "Scoring Crédit Production V2.0",
        "model": type(model).__name__ if model_loaded else "Non chargé",
        "num_features": len(feature_columns),
        "status": "OK" if model_loaded else "ERROR"
    }

@app.get("/status")
def status():
    return {
        "status": "operational" if model_loaded else "error",
        "model_loaded": model_loaded
    }

@app.get("/model/info")
def model_info():
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    return {
        "model_type": type(model).__name__,
        "num_features": len(feature_columns),
        "optimal_threshold": float(threshold)
    }

@app.post("/predict")
def predict(data: Features):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Service non disponible")
    
    if len(data.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu: {len(feature_columns)}, Reçu: {len(data.features)}"
        )
    
    X = np.array(data.features).reshape(1, -1)
    risk_score = model.predict_proba(X)[:,1][0]
    decision = "ACCORD" if risk_score < threshold else "REFUS"
    return {"risk_score": float(risk_score), "decision": decision, "threshold": float(threshold)}

@app.post("/explain")
def explain(data: Features):
    if not model_loaded or explainer is None:
        raise HTTPException(status_code=500, detail="Service SHAP non disponible")
    
    if len(data.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu: {len(feature_columns)}, Reçu: {len(data.features)}"
        )
    
    X = np.array(data.features).reshape(1, -1)
    shap_values = explainer.shap_values(X)
    
    # Pour LightGBM binaire : shap_values[1] correspond à la classe positive
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]
    
    # Tri par importance absolue et top 10 features
    feature_impact = list(zip(feature_columns, shap_vals, data.features))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_features = []
    for feat, impact, value in feature_impact[:10]:
        top_features.append({
            "feature": feat,
            "impact": float(impact),
            "value": float(value),
            "direction": "AUGMENTE LE RISQUE" if impact > 0 else "DIMINUE LE RISQUE"
        })
    
    return {"top_features": top_features, "interpretation": "Impact positif = augmente le risque | Impact négatif = diminue le risque"}

# --- POINT D'ENTRÉE POUR EXÉCUTION LOCALE ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
