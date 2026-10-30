"""
Security tests for path traversal prevention and filesystem boundary enforcement.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import PathTraversalError
from app.core.security import resolve_safe_path
from app.main import app

client = TestClient(app)


def test_path_traversal_detection():
    """Verify attempting to break out of base dir raises PathTraversalError."""
    base_dir = Path("/tmp/safe_root")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Valid subpath
    valid = resolve_safe_path(base_dir, "dataset/images/p1.png")
    assert str(valid).startswith(str(base_dir.resolve()))

    # Malicious traversal attempts
    with pytest.raises(PathTraversalError):
        resolve_safe_path(base_dir, "../../etc/passwd")

    with pytest.raises(PathTraversalError):
        resolve_safe_path(base_dir, "../../../root/.ssh/id_rsa")

    with pytest.raises(PathTraversalError):
        resolve_safe_path(base_dir, "/etc/shadow")


def test_api_rejects_path_traversal_in_image_ref():
    """Verify /v1/recognize rejects path traversal attempts."""
    response = client.post(
        "/v1/recognize",
        data={
            "image_ref": "../../etc/passwd",
            "language": "PERSIAN",
            "model_id": "demo-persian-naskh-v1",
        },
    )
    # Must reject with 403 Forbidden
    assert response.status_code == 403
    data = response.json()
    assert data["error_code"] == "PATH_TRAVERSAL_DETECTED"
