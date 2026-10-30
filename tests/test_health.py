"""
Tests for health and readiness endpoints.
Verifies separation between API server status and model readiness.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 with service version and ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["ready"] is True
    assert "timestamp" in data


def test_ready_endpoint():
    """Verify GET /ready returns dependency breakdown and model readiness state."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "service_status" in data
    assert "dependencies" in data
    assert "default_model_status" in data
    deps = data["dependencies"]
    assert "pytorch" in deps
    assert "storage" in deps
    assert "default_model" in deps
