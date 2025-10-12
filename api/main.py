from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="API Scoring Crédit - Production",
    description="API avec champion model (sans données clients)",
    version="1.0"
)

MODEL_DIR = "../models"
# Chargement des modèles
try:
    model = joblib.load(f"{MODEL_DIR}/champion_model.pkl")
    threshold = joblib.load(f"{MODEL_DIR}/champion_threshold.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    metadata = joblib.load(f"{MODEL_DIR}/model_metadata.pkl")
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

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Service non disponible")
    
    if len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400, 
            detail=f"Nombre de features incorrect. Attendu: {len(feature_columns)}, Reçu: {len(request.features)}"
        )
    
    try:
        features_array = np.array(request.features).reshape(1, -1)
        proba = model.predict_proba(features_array)[0, 1]
        decision = "REFUS" if proba >= threshold else "ACCORD"
        
        return {
            "risk_score": float(proba),
            "risk_percentage": f"{proba*100:.2f}%",
            "threshold": float(threshold),
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_prediction(request: PredictionRequest):
    """Retourne les features les plus importantes pour cette prédiction"""
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Service non disponible")
    
    if len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400, 
            detail=f"Nombre de features incorrect"
        )
    
    try:
        import shap
        explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
        
        features_array = np.array(request.features).reshape(1, -1)
        shap_values = explainer.shap_values(features_array)
        
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
        raise HTTPException(status_code=500, detail=f"Erreur SHAP: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)