import requests

url = "https://api-scoring-credit-final.onrender.com/explain"

# Crée un vecteur de 254 features pour tester l'API
features = [0]*254

data = {"features": features}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
