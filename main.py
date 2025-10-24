from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="API Scoring Crédit - Production",
    description="API avec champion model (sans données clients)",
    version="1.0"
)

MODEL_DIR = "../models"

# Chargement des modèles
try:
    model = joblib.load(os.path.join(MODEL_DIR, "champion_model.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "champion_threshold.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "model_metadata.pkl"))
    print(f"✓ Modèle: {metadata['model_type']}, Seuil: {threshold:.3f}, Features: {len(feature_columns)}")
    model_loaded = True
except Exception as e:
    print(f"Erreur modèle: {e}")
    model_loaded = False

class PredictionRequest(BaseModel):
    features: list[float]

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
    
    if len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu: {len(feature_columns)}, Reçu: {len(request.features)}"
        )
    
    features_array = np.array(request.features).reshape(1, -1)
    proba = model.predict_proba(features_array)[0, 1]
    decision = "REFUS" if proba >= threshold else "ACCORD"
    
    return {
        "risk_score": float(proba),
        "decision": decision,
        "threshold": float(threshold)
    }

@app.post("/explain")
def explain_prediction(request: PredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Service non disponible")
    
    if len(request.features) != len(feature_columns):
        raise HTTPException(status_code=400, detail=f"Nombre de features incorrect")
    
    try:
        import shap

        explainer_path = os.path.join(MODEL_DIR, "shap_explainer.pkl")
        if not os.path.exists(explainer_path):
            raise FileNotFoundError(f"{explainer_path} non trouvé sur le serveur")

        explainer = joblib.load(explainer_path)
        features_array = np.array(request.features).reshape(1, -1)
        print("DEBUG: Features shape:", features_array.shape)

        shap_values = explainer.shap_values(features_array)
        print("DEBUG: SHAP values type:", type(shap_values))

        # Compatibilité LightGBM binaire vs multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]

        feature_impact = list(zip(feature_columns, shap_values, request.features))
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
        print("Erreur SHAP:", str(e))
        raise HTTPException(status_code=500, detail=f"Erreur SHAP: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
