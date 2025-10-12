"""
Tests unitaires pour l'API Scoring Crédit - coverage améliorée
"""

import pytest
from fastapi.testclient import TestClient
from main import app, model_loaded, MODEL_DIR
import joblib
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
    assert data["model"] in ["lightgbm", "Non chargé"]
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
        assert data["model_type"] == "lightgbm"
    else:
        # Simuler le cas modèle non chargé
        response = client.get("/model/info")
        assert response.status_code == 500


# -------------------------
# Tests predict
# -------------------------

def test_predict_valid_features():
    if not model_loaded:
        pytest.skip("Modèle non chargé, test ignoré")
    features = [0.5] * 254
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "decision" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["decision"] in ["ACCORD", "REFUS"]


def test_predict_invalid_feature_count():
    features = [0.5] * 100
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 400


def test_predict_model_not_loaded(monkeypatch):
    # Simuler un modèle non chargé
    monkeypatch.setattr("main.model_loaded", False)
    features = [0.5] * 254
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 500


def test_predict_deterministic():
    if not model_loaded:
        pytest.skip("Modèle non chargé, test ignoré")
    features = [0.7] * 254
    response1 = client.post("/predict", json={"features": features})
    response2 = client.post("/predict", json={"features": features})
    assert response1.json()["risk_score"] == response2.json()["risk_score"]


# -------------------------
# Tests explain_prediction
# -------------------------

def test_explain_invalid_feature_count():
    features = [0.5] * 200
    response = client.post("/explain", json={"features": features})
    assert response.status_code == 400


def test_explain_valid_features(monkeypatch):
    if not model_loaded:
        pytest.skip("Modèle non chargé, test ignoré")

    # On simule un explainer pour éviter les erreurs SHAP
    class FakeExplainer:
        def shap_values(self, X):
            return [np.zeros(X.shape), np.zeros(X.shape)]

    monkeypatch.setattr("main.joblib.load", lambda path: FakeExplainer())
    features = [0.5] * 254
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
