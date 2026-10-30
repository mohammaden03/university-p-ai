"""
Tests for dataset manifest validation, missing file detection, and integrity checks.
"""

import json

import pytest

from app.core.config import settings
from app.core.exceptions import DatasetValidationError
from app.services.dataset_service import DatasetService


def test_validate_real_sample_manifest():
    """Verify our generated sample manifest passes validation."""
    manifest_rel = "sample_persian_arabic/manifest.jsonl"
    result = DatasetService.validate_dataset(manifest_rel, verify_images=True)
    assert result["valid"] is True
    assert result["total_samples"] >= 10
    assert result["split_counts"]["train"] >= 4
    assert result["split_counts"]["test"] >= 2


def test_manifest_missing_image_fails():
    """Verify missing referenced image raises DatasetValidationError."""
    temp_manifest = settings.DATASET_ROOT / "temp_invalid_img.jsonl"
    bad_record = {
        "id": "bad_sample_1",
        "image_path": "sample_persian_arabic/images/nonexistent_file_99999.png",
        "text": "متن نمونه",
        "split": "train",
    }
    with open(temp_manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps(bad_record) + "\n")

    try:
        with pytest.raises(DatasetValidationError) as exc:
            DatasetService.validate_dataset("temp_invalid_img.jsonl", verify_images=True)
        assert "Image file does not exist" in str(exc.value)
    finally:
        if temp_manifest.exists():
            temp_manifest.unlink()


def test_manifest_empty_text_fails():
    """Verify empty ground-truth transcription raises DatasetValidationError."""
    temp_manifest = settings.DATASET_ROOT / "temp_empty_text.jsonl"
    bad_record = {
        "id": "bad_sample_2",
        "image_path": "sample_persian_arabic/images/ms_golestan_001_l1.png",
        "text": "   ",  # only whitespace
        "split": "train",
    }
    with open(temp_manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps(bad_record) + "\n")

    try:
        with pytest.raises(DatasetValidationError) as exc:
            DatasetService.validate_dataset("temp_empty_text.jsonl", verify_images=False)
        assert "Empty ground-truth transcription" in str(exc.value)
    finally:
        if temp_manifest.exists():
            temp_manifest.unlink()
