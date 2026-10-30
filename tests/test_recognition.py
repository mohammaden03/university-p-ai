"""
Tests for /v1/recognize endpoint.
Verifies multipart upload, internal references, language support, and schema adherence.
"""

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def get_sample_image_file() -> tuple:
    """Helper to generate in-memory file for multipart upload."""
    im = Image.new("RGB", (300, 70), color=(245, 238, 220))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return ("manuscript.png", buf.getvalue(), "image/png")


def test_recognize_demo_engine():
    """Verify recognition with demo engine returns valid schema and lines."""
    fname, content, ctype = get_sample_image_file()
    response = client.post(
        "/v1/recognize",
        files={"file": (fname, content, ctype)},
        data={
            "language": "PERSIAN",
            "model_id": "demo-persian-naskh-v1",
            "layout_analysis": "true",
            "contrast": "true",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "request_id" in data
    assert len(data["text"]) > 0
    assert data["language"] == "PERSIAN"
    assert data["model"]["id"] == "demo-persian-naskh-v1"
    assert data["processing_time_ms"] > 0
    assert len(data["lines"]) >= 1
    # Check bounding box
    box = data["lines"][0]["bbox"]
    assert box["x"] >= 0 and box["y"] >= 0 and box["w"] > 0 and box["h"] > 0


def test_recognize_internal_image_ref():
    """Verify recognition with internal image_ref parameter."""
    response = client.post(
        "/v1/recognize",
        data={
            "image_ref": "sample_persian_arabic/images/ms_golestan_001_l1.png",
            "language": "ARABIC",
            "model_id": "demo-persian-naskh-v1",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "بسم الله" in data["text"]


def test_recognize_invalid_language():
    """Verify requesting an unsupported language returns 422 validation error."""
    fname, content, ctype = get_sample_image_file()
    response = client.post(
        "/v1/recognize",
        files={"file": (fname, content, ctype)},
        data={"language": "SPANISH"},
    )
    assert response.status_code == 422


def test_recognize_missing_both_image_and_ref():
    """Verify omitting both file and image_ref returns 400."""
    response = client.post(
        "/v1/recognize",
        data={"language": "PERSIAN"},
    )
    assert response.status_code == 400
