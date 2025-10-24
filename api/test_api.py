"""
Tests unitaires pour l'API Scoring Crédit - coverage améliorée
"""

import pytest
from fastapi.testclient import TestClient
from main import app, model_loaded, feature_columns, threshold
import numpy as np

client = TestClient(app)

# -------------------------
# Tests endpoints principaux
# -------------------------

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["OK", "ERROR"]
    assert "model" in data
    assert isinstance(data["num_features"], int)

def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["operational", "error"]
    assert "model_loaded" in data

def test_model_info_endpoint():
    if model_loaded:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "auc_score" in data
        assert "optimal_threshold" in data
    else:
        response = client.get("/model/info")
        assert response.status_code == 500

# -------------------------
# Tests predict
# -------------------------

def test_predict_valid_features():
    if not model_loaded:
        pytest.skip("Modèle non chargé, test ignoré")
    features = [0.5] * len(feature_columns)
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "decision" in data
    assert "threshold" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["decision"] in ["ACCORD", "REFUS"]
    assert data["threshold"] == threshold

def test_predict_invalid_feature_count():
    features = [0.5] * (len(feature_columns) - 10)
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 400

def test_predict_model_not_loaded(monkeypatch):
    monkeypatch.setattr("main.model_loaded", False)
    features = [0.5] * len(feature_columns)
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 500

def test_predict_deterministic():
    if not model_loaded:
        pytest.skip("Modèle non chargé, test ignoré")
    features = [0.7] * len(feature_columns)
    response1 = client.post("/predict", json={"features": features})
    response2 = client.post("/predict", json={"features": features})
    assert response1.json()["risk_score"] == response2.json()["risk_score"]

# -------------------------
# Tests explain_prediction
# -------------------------

def test_explain_invalid_feature_count():
    features = [0.5] * (len(feature_columns) - 20)
    response = client.post("/explain", json={"features": features})
    assert response.status_code == 400

def test_explain_valid_features(monkeypatch):
    if not model_loaded:
        pytest.skip("Modèle non chargé, test ignoré")

    # Simulation d'un explainer pour éviter des erreurs SHAP
    class FakeExplainer:
        def shap_values(self, X):
            return [np.zeros(X.shape), np.zeros(X.shape)]

    monkeypatch.setattr("main", "explainer", FakeExplainer())

    features = [0.5] * len(feature_columns)
    response = client.post("/explain", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) == 10
    for feat in data["top_features"]:
        assert "feature" in feat
        assert "impact" in feat
        assert "value" in feat
        assert "direction" in feat
