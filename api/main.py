from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path

# --- INITIALISATION DE L'API ---
app = FastAPI(
    title="API Scoring Crédit - Production",
    description="API avec champion model et seuil optimal",
    version="2.0"
)

# --- CHEMIN DU RÉPERTOIRE MODELS ---
MODEL_DIR = Path(__file__).parent / "models"

# --- VARIABLES GLOBALES ---
model_loaded = False
model = None
threshold = None
feature_columns = None
metadata = None
explainer = None

# --- CHARGEMENT DU MODÈLE AU DÉMARRAGE ---
try:
    model_path = MODEL_DIR / "champion_model.pkl"
    threshold_path = MODEL_DIR / "champion_threshold.pkl"
    feature_columns_path = MODEL_DIR / "feature_columns.pkl"
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    shap_explainer_path = MODEL_DIR / "shap_explainer.pkl"

    for f in [model_path, threshold_path, feature_columns_path, metadata_path, shap_explainer_path]:
        if not f.exists():
            raise FileNotFoundError(f"Fichier introuvable : {f}")

    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    feature_columns = joblib.load(feature_columns_path)
    metadata = joblib.load(metadata_path)
    explainer = joblib.load(shap_explainer_path)

    model_loaded = True
    print(f"✓ Modèle chargé : {metadata['model_type']}, seuil={threshold}, features={len(feature_columns)}")

except Exception as e:
    print(f"Erreur au chargement du modèle : {e}")
    model_loaded = False

# --- SCHÉMA DES FEATURES ---
class PredictionRequest(BaseModel):
    features: List[float]

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {
        "api": "Scoring Crédit Production V2.0",
        "model": metadata['model_type'] if model_loaded else "Non chargé",
        "num_features": len(feature_columns) if model_loaded else 0,
        "status": "OK" if model_loaded else "ERROR"
    }

@app.get("/status")
def health_check():
    return {
        "status": "operational" if model_loaded else "error",
        "model_loaded": model_loaded
    }

@app.get("/model/info")
def model_info():
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    return {
        "model_type": metadata['model_type'],
        "auc_score": float(metadata['auc_score']),
        "optimal_threshold": float(threshold),
        "optimal_cost": float(metadata['optimal_cost']),
        "num_features": len(feature_columns),
        "training_date": metadata['training_date']
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Service non disponible")
    if feature_columns is None or len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu: {len(feature_columns) if feature_columns else 'inconnu'}, Reçu: {len(request.features)}"
        )
    try:
        features_array = np.array(request.features).reshape(1, -1)
        proba = model.predict_proba(features_array)[0, 1]
        decision = "REFUS" if proba >= threshold else "ACCORD"
        return {
            "risk_score": float(proba),
            "decision": decision,
            "threshold": float(threshold)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prediction: {str(e)}")

@app.post("/explain")
def explain_prediction(request: PredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Service non disponible")
    if feature_columns is None or len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu: {len(feature_columns) if feature_columns else 'inconnu'}, Reçu: {len(request.features)}"
        )
    try:
        features_array = np.array(request.features).reshape(1, -1)
        shap_values = explainer.shap_values(features_array)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # classe positive
        else:
            shap_vals = shap_values[0]

        feature_impact = list(zip(feature_columns, shap_vals, request.features))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for feat, impact, value in feature_impact[:10]:
            top_features.append({
                "feature": feat,
                "impact": float(impact),
                "value": float(value),
                "direction": "AUGMENTE LE RISQUE" if impact > 0 else "DIMINUE LE RISQUE"
            })

        return {
            "top_features": top_features,
            "interpretation": "Impact positif = augmente le risque de défaut | Impact négatif = diminue le risque"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP: {str(e)}")

# --- POUR LANCER LOCALEMENT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
