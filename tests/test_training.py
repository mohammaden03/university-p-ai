"""
Tests for model training workflows and asynchronous job tracking.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_start_training_job_success():
    """Verify POST /v1/train returns 202 with job_id for valid manifests."""
    payload = {
        "model_name": "test-persian-crnn",
        "train_manifest": "sample_persian_arabic/train.jsonl",
        "val_manifest": "sample_persian_arabic/val.jsonl",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "random_seed": 42,
    }
    response = client.post("/v1/train", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["model_name"] == "test-persian-crnn"
    assert data["status"] in ["PENDING", "RUNNING"]

    job_id = data["job_id"]
    # Poll status
    status_resp = client.get(f"/v1/train/{job_id}")
    assert status_resp.status_code == 200
    st_data = status_resp.json()
    assert st_data["job_id"] == job_id


def test_start_training_invalid_manifest_rejected():
    """Verify training rejects non-existent dataset manifests immediately."""
    payload = {
        "model_name": "invalid-manifest-model",
        "train_manifest": "sample_persian_arabic/does_not_exist.jsonl",
        "val_manifest": "sample_persian_arabic/val.jsonl",
        "epochs": 1,
    }
    response = client.post("/v1/train", json=payload)
    # Must fail with 500 / 422 / 404
    assert response.status_code in [404, 422, 500]
