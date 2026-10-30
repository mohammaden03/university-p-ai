"""
Tests for image preprocessing pipeline and validation.
"""

import io

import numpy as np
import pytest
from PIL import Image

from app.core.exceptions import InvalidImageError
from app.services.preprocessing_service import PreprocessingService


def create_test_image_bytes(width=200, height=80, format="PNG") -> bytes:
    """Helper to generate in-memory image bytes."""
    im = Image.new("RGB", (width, height), color=(240, 230, 210))
    buf = io.BytesIO()
    im.save(buf, format=format)
    return buf.getvalue()


def test_decode_valid_image():
    """Verify decoding a valid image returns a PIL Image."""
    raw_bytes = create_test_image_bytes(200, 80, "PNG")
    img = PreprocessingService.decode_and_validate_image(raw_bytes, "test.png")
    assert isinstance(img, Image.Image)
    assert img.size == (200, 80)


def test_decode_empty_bytes_fails():
    """Verify empty image payload raises InvalidImageError."""
    with pytest.raises(InvalidImageError):
        PreprocessingService.decode_and_validate_image(b"", "empty.jpg")


def test_decode_corrupted_image_fails():
    """Verify random corrupted bytes raise InvalidImageError."""
    corrupted = b"NOT_A_VALID_IMAGE_HEADER_1234567890"
    with pytest.raises(InvalidImageError):
        PreprocessingService.decode_and_validate_image(corrupted, "corrupt.jpg")


def test_binarize_otsu():
    """Verify Otsu binarization produces binary image (values 0 or 255)."""
    gray = Image.new("L", (100, 100), color=120)
    # Add dark region
    for x in range(30, 70):
        for y in range(30, 70):
            gray.putpixel((x, y), 20)

    binary = PreprocessingService.binarize_otsu(gray)
    arr = np.array(binary)
    unique_vals = set(np.unique(arr))
    assert unique_vals.issubset({0, 255})


def test_process_pipeline_preserves_original():
    """Verify preprocessing operations return new image without mutating original."""
    raw_bytes = create_test_image_bytes(150, 50, "JPEG")
    original = PreprocessingService.decode_and_validate_image(raw_bytes)
    orig_mode = original.mode

    processed, meta = PreprocessingService.process_pipeline(
        original,
        grayscale=True,
        deskew=True,
        contrast=True,
        denoise=True,
        binarize=True,
    )

    # Processed image is binary/grayscale
    assert processed.mode == "L"
    # Original remains in RGB
    assert original.mode == orig_mode
    assert len(meta["steps_applied"]) > 0
