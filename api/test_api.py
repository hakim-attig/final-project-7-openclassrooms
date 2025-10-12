"""
Tests unitaires pour l'API Scoring Crédit
Exécuté automatiquement par GitHub Actions
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """Teste que l'endpoint racine retourne le statut OK"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["model"] == "lightgbm"
    assert data["num_features"] == 254


def test_status_endpoint():
    """Teste le health check"""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_model_info_endpoint():
    """Teste les métadonnées du modèle"""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "auc" in data
    assert data["model_type"] == "lightgbm"


def test_predict_valid_features():
    """Teste une prédiction avec 254 features valides"""
    features = [0.5] * 254
    response = client.post("/predict", json={"features": features})
    
    assert response.status_code == 200
    data = response.json()
    
    assert "risk_score" in data
    assert "decision" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["decision"] in ["ACCORD", "REFUS"]


def test_predict_invalid_feature_count():
    """Teste que l'API rejette un nombre incorrect de features"""
    features = [0.5] * 100
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 400


def test_explain_valid_features():
    """Teste l'explication SHAP avec features valides"""
    features = [0.5] * 254
    response = client.post("/explain", json={"features": features})
    
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) == 10


def test_explain_invalid_feature_count():
    """Teste que explain rejette un nombre incorrect de features"""
    features = [0.5] * 200
    response = client.post("/explain", json={"features": features})
    assert response.status_code == 400


def test_predict_deterministic():
    """Vérifie que le modèle est déterministe"""
    features = [0.7] * 254
    
    response1 = client.post("/predict", json={"features": features})
    response2 = client.post("/predict", json={"features": features})
    
    assert response1.json()["risk_score"] == response2.json()["risk_score"]