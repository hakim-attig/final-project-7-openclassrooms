from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="API Scoring Crédit - Production",
    description="API avec calcul SHAP côté serveur (modèle champion)",
    version="2.0"
)

MODEL_DIR = "../models"

# Chargement des modèles et objets
try:
    model = joblib.load(f"{MODEL_DIR}/champion_model.pkl")
    threshold = joblib.load(f"{MODEL_DIR}/champion_threshold.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    metadata = joblib.load(f"{MODEL_DIR}/model_metadata.pkl")
    explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")

    print(f"✓ Modèle: {metadata['model_type']}, Seuil: {threshold:.3f}, Features: {len(feature_columns)}")
    model_loaded = True
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model_loaded = False

# ======== Schéma d’entrée =========
class PredictionRequest(BaseModel):
    features: list[float]

# ======== Endpoints ===============
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
    """Renvoie la prédiction, la décision et les valeurs SHAP du modèle."""
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Service non disponible")
    
    if len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de features incorrect. Attendu: {len(feature_columns)}, Reçu: {len(request.features)}"
        )

    try:
        # --- Prédiction du risque ---
        features_array = np.array(request.features).reshape(1, -1)
        proba = model.predict_proba(features_array)[0, 1]
        decision = "REFUS" if proba >= threshold else "ACCORD"

        # --- Valeurs SHAP ---
        import shap
        shap_values = explainer.shap_values(features_array)
        if isinstance(shap_values, list):  # cas modèle binaire
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]

        feature_impact = list(zip(feature_columns, shap_values, request.features))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for feat, impact, value in feature_impact[:20]:
            top_features.append({
                "feature": feat,
                "impact": float(impact),
                "value": float(value),
                "direction": "AUGMENTE LE RISQUE" if impact > 0 else "DIMINUE LE RISQUE"
            })

        # --- Valeur de base SHAP ---
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        # --- Réponse finale ---
        return {
            "risk_score": float(proba),
            "decision": decision,
            "threshold": float(threshold),
            "top_features": top_features,
            "shap_values": [float(v) for v in shap_values],
            "feature_names": feature_columns,
            "base_value": float(base_value)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction/SHAP: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
