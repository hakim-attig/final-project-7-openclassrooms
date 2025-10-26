from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path
import shap
import sys

# --- INITIALISATION DE L'API ---
app = FastAPI(
    title="API Scoring Crédit - Production",
    description="API avec champion model et seuil optimal",
    version="2.0"
)

# --- CHEMIN DU RÉPERTOIRE MODELS ---
# Supporte local et Render
MODEL_DIR = Path(__file__).parent / "models"

print("="*70)
print("🚀 DÉMARRAGE DE L'API")
print("="*70)
print("MODEL_DIR:", MODEL_DIR)
print(f"Python version: {sys.version.split()[0]}")

if MODEL_DIR.exists():
    print("\n📁 Fichiers dans le dossier models:")
    for f in sorted(MODEL_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
else:
    print("❌ Dossier models introuvable !")

# --- VARIABLES GLOBALES ---
model_loaded = False
model = None
threshold = None
feature_columns = None
metadata = None
explainer = None

# --- CHARGEMENT DU MODÈLE AU DÉMARRAGE ---
print("\n" + "="*70)
print("📦 CHARGEMENT DES MODÈLES")
print("="*70)

try:
    model_path = MODEL_DIR / "champion_model.pkl"
    threshold_path = MODEL_DIR / "champion_threshold.pkl"
    feature_columns_path = MODEL_DIR / "feature_columns.pkl"
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    shap_explainer_path = MODEL_DIR / "shap_explainer.pkl"

    # Vérifier que tous les fichiers existent
    missing_files = [f.name for f in [model_path, threshold_path, feature_columns_path, metadata_path, shap_explainer_path] if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Fichiers manquants : {missing_files}")

    # 1. Charger le modèle
    print("\n1️⃣ Chargement du modèle champion...")
    model = joblib.load(model_path)
    print(f"   ✓ Type: {type(model).__name__}")

    # 2. Charger le seuil
    print("\n2️⃣ Chargement du seuil optimal...")
    threshold = joblib.load(threshold_path)
    print(f"   ✓ Seuil: {threshold:.3f}")

    # 3. Charger les features
    print("\n3️⃣ Chargement des features...")
    feature_columns = joblib.load(feature_columns_path)
    print(f"   ✓ Nombre de features: {len(feature_columns)}")

    # 4. Charger les métadonnées
    print("\n4️⃣ Chargement des métadonnées...")
    metadata = joblib.load(metadata_path)
    print(f"   ✓ Modèle: {metadata['model_type']}")
    print(f"   ✓ AUC: {metadata['auc_score']:.4f}")

    # 5. Charger l'explainer SHAP (AVEC GESTION D'ERREUR)
    print("\n5️⃣ Chargement de l'explainer SHAP...")
    try:
        # Essayer avec dill d'abord (meilleure compatibilité)
        try:
            import dill
            with open(shap_explainer_path, "rb") as f:
                explainer = dill.load(f)
            print(f"   ✓ Chargé avec dill: {type(explainer).__name__}")
        except (ImportError, Exception) as e:
            # Fallback sur joblib
            explainer = joblib.load(shap_explainer_path)
            print(f"   ✓ Chargé avec joblib: {type(explainer).__name__}")
    
    except Exception as e:
        # Si échec, recréer l'explainer
        print(f"   ⚠️  Échec du chargement: {str(e)[:80]}...")
        print(f"   🔄 Recréation de l'explainer...")
        explainer = shap.TreeExplainer(model)
        print(f"   ✓ Explainer recréé: {type(explainer).__name__}")

    model_loaded = True
    print("\n" + "="*70)
    print("✅ TOUS LES MODÈLES CHARGÉS AVEC SUCCÈS")
    print("="*70)
    print(f"Modèle: {metadata['model_type']}")
    print(f"Seuil: {threshold:.3f}")
    print(f"Features: {len(feature_columns)}")
    print(f"SHAP: {'✓ Disponible' if explainer else '✗ Indisponible'}")
    print("="*70 + "\n")

except Exception as e:
    print("\n" + "="*70)
    print("❌ ERREUR CRITIQUE AU CHARGEMENT")
    print("="*70)
    print(f"Erreur: {e}")
    import traceback
    print(traceback.format_exc())
    print("="*70 + "\n")
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
        "status": "OK" if model_loaded else "ERROR",
        "shap_available": explainer is not None
    }

@app.get("/status")
def health_check():
    return {
        "status": "operational" if model_loaded else "error",
        "model_loaded": model_loaded,
        "shap_loaded": explainer is not None
    }

@app.get("/model/info")
def model_info():
    if not model_loaded or metadata is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    return {
        "model_type": metadata['model_type'],
        "auc_score": float(metadata['auc_score']),
        "optimal_threshold": float(threshold),
        "optimal_cost": float(metadata['optimal_cost']),
        "num_features": len(feature_columns),
        "training_date": metadata['training_date'],
        "shap_available": explainer is not None
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model_loaded or feature_columns is None:
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
    if not model_loaded or feature_columns is None:
        raise HTTPException(status_code=500, detail="Service non disponible")
    
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer SHAP non disponible")

    if len(request.features) != len(feature_columns):
        raise HTTPException(status_code=400, detail="Nombre de features incorrect")

    features_array = np.array(request.features).reshape(1, -1)
    
    # Calculer les SHAP values (compatible avec les anciennes et nouvelles versions)
    try:
        # Nouvelle API SHAP (0.40+)
        shap_explanation = explainer(features_array)
        if hasattr(shap_explanation, 'values'):
            if len(shap_explanation.values.shape) == 3:
                shap_vals = shap_explanation.values[0, :, 1]  # classe positive
            else:
                shap_vals = shap_explanation.values[0]
        else:
            shap_vals = shap_explanation[0]
    except:
        # Ancienne API SHAP
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

# --- POUR LANCER LOCALEMENT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)