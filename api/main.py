from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="API Scoring Crédit - Production",
    description="API avec modèle champion et seuil optimal",
    version="2.0"
)

# Répertoire contenant les fichiers du modèle
MODEL_DIR = "models"  # adapte à ton cas ("../models" si besoin)

# Chargement conditionnel du modèle
model_loaded = False
model = None
threshold = None
feature_columns = []
metadata = {}

# --- Chargement du modèle uniquement si les fichiers existent ---
required_files = [
    "champion_model.pkl",
    "champion_threshold.pkl",
    "feature_columns.pkl",
    "model_metadata.pkl"
]

missing = [f for f in required_files if not os.path.exists(f"{MODEL_DIR}/{f}")]
if missing:
    print(f"⚠️ Fichiers manquants : {missing}")
    print("→ L’API démarre en mode dégradé (sans modèle).")
else:
    try:
        model = joblib.load(f"{MODEL_DIR}/champion_model.pkl")
        threshold = joblib.load(f"{MODEL_DIR}/champion_threshold.pkl")
        feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
        metadata = joblib.load(f"{MODEL_DIR}/model_metadata.pkl")

        print(f"✓ Modèle chargé : {metadata['model_type']} | Seuil optimal : {threshold:.3f}")
        model_loaded = True
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")

# --- Schéma de données pour les prédictions ---
class PredictionRequest(BaseModel):
    features: list[float]

# --- Endpoints ---
@app.get("/")
def root():
    """Informations générales sur l’API"""
    return {
        "api": "Scoring Crédit Production V2.0",
        "model": metadata.get('model_type', 'Non chargé'),
        "num_features": len(feature_columns),
        "threshold": float(threshold) if threshold else None,
        "status": "OK" if model_loaded else "MODE DÉGRADÉ"
    }

@app.get("/status")
def health_check():
    """Vérifie la disponibilité du modèle"""
    return {
        "status": "operational" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }

@app.get("/model/info")
def model_info():
    """Retourne les métadonnées du modèle"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
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
    """Retourne la probabilité de risque et la décision"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    if len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect : {len(request.features)} reçues, {len(feature_columns)} attendues."
        )

    X = np.array(request.features).reshape(1, -1)
    proba = model.predict_proba(X)[0, 1]
    decision = "REFUS" if proba >= threshold else "ACCORD"

    return {
        "risk_score": float(proba),
        "decision": decision,
        "threshold": float(threshold)
    }

@app.post("/explain")
def explain_prediction(request: PredictionRequest):
    """Explique la prédiction via SHAP"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        import shap
        explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
        X = np.array(request.features).reshape(1, -1)
        shap_values = explainer.shap_values(X)
        shap_values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        feature_impact = sorted(
            zip(feature_columns, shap_values, request.features),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = [
            {
                "feature": feat,
                "impact": float(impact),
                "value": float(value),
                "direction": "AUGMENTE LE RISQUE" if impact > 0 else "DIMINUE LE RISQUE"
            }
            for feat, impact, value in feature_impact[:10]
        ]

        return {
            "top_features": top_features,
            "interpretation": "Impact positif = augmente le risque de défaut | Impact négatif = diminue le risque"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
