import requests
import json

API_URL = "http://127.0.0.1:8000"  # Adresse locale de ton API

# Exemple : générer un vecteur de features avec la bonne taille
# Remplace 254 par le nombre exact de features dans ton modèle
features = [0.0] * 254  

# 1️⃣ Test endpoint /predict
predict_resp = requests.post(f"{API_URL}/predict", json={"features": features})
print("=== /predict ===")
print("Status code:", predict_resp.status_code)
print("Response JSON:", json.dumps(predict_resp.json(), indent=4))

# 2️⃣ Test endpoint /explain
explain_resp = requests.post(f"{API_URL}/explain", json={"features": features})
print("\n=== /explain ===")
print("Status code:", explain_resp.status_code)
print("Response JSON:", json.dumps(explain_resp.json(), indent=4))
