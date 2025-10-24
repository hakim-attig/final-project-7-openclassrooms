import joblib
import shap
import numpy as np

# --- CHARGEMENT DU MODÈLE ---
model_path = "models/champion_model.pkl"
model = joblib.load(model_path)
print("✅ Modèle chargé avec succès :", type(model))

# --- CRÉATION DE L'EXPLAINER ---
try:
    explainer = shap.TreeExplainer(model)
    print("✅ Explainer SHAP créé avec succès.")
except Exception as e:
    print("❌ Erreur lors de la création de l'explainer :", e)

# --- TEST AVEC UN EXEMPLE ALÉATOIRE ---
X = np.random.rand(1, 254)  # 254 features
try:
    shap_values = explainer.shap_values(X)
    print("✅ SHAP values calculées avec succès.")
    print("Forme du résultat :", np.array(shap_values).shape)
except Exception as e:
    print("❌ Erreur lors du calcul des SHAP values :", e)
