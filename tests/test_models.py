"""
Tests for Model Registry and /v1/models endpoints.
Verifies honest state reporting and model discovery.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.models import ModelStatus

client = TestClient(app)


def test_list_models():
    """Verify /v1/models returns registered models and capabilities."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "default_model_id" in data
    assert "models" in data
    assert len(data["models"]) >= 3

    model_ids = [m["id"] for m in data["models"]]
    assert "persian-arabic-crnn-v1" in model_ids
    assert "kraken-arabic-persian-v1" in model_ids
    assert "demo-persian-naskh-v1" in model_ids


def test_get_specific_model():
    """Verify /v1/models/{model_id} returns detailed metadata."""
    response = client.get("/v1/models/persian-arabic-crnn-v1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "persian-arabic-crnn-v1"
    assert "PERSIAN" in data["supported_languages"]
    assert "ARABIC" in data["supported_languages"]
    assert "capabilities" in data
    assert data["status"] in [s.value for s in ModelStatus]


def test_get_nonexistent_model():
    """Verify requesting an unknown model ID returns 404."""
    response = client.get("/v1/models/unknown-model-xyz")
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "MODEL_NOT_FOUND"
