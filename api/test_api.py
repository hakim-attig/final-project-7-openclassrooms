import pytest
from fastapi.testclient import TestClient
from main import app, model_loaded, feature_columns

client = TestClient(app)

# --- TEST DE BASE : API UP ---
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert "status" in data
    assert data["status"] in ["OK", "MODE DÉGRADÉ"]

# --- TEST /status ---
def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)

# --- TEST /model/info ---
def test_model_info_endpoint():
    response = client.get("/model/info")
    if model_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "optimal_threshold" in data
    else:
        assert response.status_code == 503  # car modèle non chargé

# --- TEST /predict ---
def test_predict_endpoint():
    # Mock features (longueur correcte si possible, sinon petit échantillon)
    n_features = len(feature_columns) if feature_columns else 10
    dummy_features = [0.0] * n_features

    response = client.post("/predict", json={"features": dummy_features})
    if model_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "decision" in data
        assert "threshold" in data
    else:
        assert response.status_code == 503

# --- TEST /explain ---
def test_explain_endpoint():
    n_features = len(feature_columns) if feature_columns else 10
    dummy_features = [0.0] * n_features

    response = client.post("/explain", json={"features": dummy_features})
    if model_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "top_features" in data
        assert isinstance(data["top_features"], list)
    else:
        # Le modèle n’est pas chargé → erreur 503
        assert response.status_code == 503
