"""
Configurable image preprocessing pipeline for historical manuscripts.
Preserves historical ink nuances while providing optional deskew, contrast,
denoise, and binarization transformations.
"""

import io
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.core.config import settings
from app.core.exceptions import ImageTooLargeError, InvalidImageError


class PreprocessingService:
    """Configurable image preprocessing service for Persian and Arabic HTR."""

    @staticmethod
    def decode_and_validate_image(
        image_bytes: bytes,
        filename: str = "upload.jpg"
    ) -> Image.Image:
        """
        Decode raw image bytes, enforce size and pixel limits, and correct EXIF orientation.
        """
        # Validate file size
        byte_size = len(image_bytes)
        if byte_size == 0:
            raise InvalidImageError("Image payload is completely empty.")
        if byte_size > settings.MAX_UPLOAD_SIZE:
            raise ImageTooLargeError(byte_size, settings.MAX_UPLOAD_SIZE, "bytes")

        # Decode image using PIL
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()  # Force reading entire image to catch truncated data
        except Exception as e:
            raise InvalidImageError(f"Failed to decode image file '{filename}': {e!s}")

        # Validate image format
        allowed_formats = {"JPEG", "JPG", "PNG", "TIFF", "TIF", "WEBP"}
        if img.format and img.format.upper() not in allowed_formats:
            raise InvalidImageError(f"Unsupported image format: {img.format}. Allowed: {allowed_formats}")

        # Check total pixel count
        w, h = img.size
        total_pixels = w * h
        if total_pixels > settings.MAX_IMAGE_PIXELS:
            raise ImageTooLargeError(total_pixels, settings.MAX_IMAGE_PIXELS, "pixels")

        # EXIF orientation correction
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass  # Ignore EXIF errors if missing

        return img

    @staticmethod
    def detect_skew_angle(gray_np: np.ndarray) -> float:
        """
        Estimate skew angle using projection profile variance.
        Tests angles from -15 to +15 degrees and finds maximum variance of horizontal projection.
        """
        h, w = gray_np.shape
        if h < 20 or w < 20:
            return 0.0

        # Downsample for fast angle search
        pil_img = Image.fromarray(gray_np)
        small = pil_img.resize((min(w, 400), min(h, 400)), Image.Resampling.NEAREST)
        arr = 255.0 - np.array(small, dtype=np.float32)

        best_angle = 0.0
        max_var = -1.0

        # Coarse sweep: -10 to +10 degrees in steps of 1 degree
        for angle in range(-10, 11, 2):
            rotated = Image.fromarray(arr).rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            proj = np.sum(np.array(rotated), axis=1)
            variance = np.var(proj)
            if variance > max_var:
                max_var = variance
                best_angle = float(angle)

        return best_angle

    @staticmethod
    def deskew_image(img: Image.Image) -> tuple[Image.Image, float]:
        """Detect and correct page skew."""
        gray = img.convert("L")
        angle = PreprocessingService.detect_skew_angle(np.array(gray))
        if abs(angle) > 0.5:
            # Rotate image with white background fill
            rotated = img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
            return rotated, angle
        return img, 0.0

    @staticmethod
    def enhance_contrast(img: Image.Image, factor: float = 1.3) -> Image.Image:
        """Enhance contrast to emphasize faded historical calligraphy ink."""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    @staticmethod
    def denoise_image(img: Image.Image) -> Image.Image:
        """Apply light median filter to suppress paper grain and scanning noise."""
        return img.filter(ImageFilter.MedianFilter(size=3))

    @staticmethod
    def binarize_otsu(gray_img: Image.Image) -> Image.Image:
        """
        Binarize grayscale image using Otsu's thresholding method on NumPy array.
        """
        arr = np.array(gray_img)
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        total = arr.size
        current_max, threshold = 0.0, 128
        sum_total = np.dot(np.arange(256), hist)

        sum_b, weight_b = 0.0, 0
        for i in range(256):
            weight_b += hist[i]
            if weight_b == 0:
                continue
            weight_f = total - weight_b
            if weight_f == 0:
                break
            sum_b += i * hist[i]
            mean_b = sum_b / weight_b
            mean_f = (sum_total - sum_b) / weight_f
            between_variance = weight_b * weight_f * (mean_b - mean_f) ** 2
            if between_variance > current_max:
                current_max = between_variance
                threshold = i

        binary = ((arr > threshold) * 255).astype(np.uint8)
        return Image.fromarray(binary)

    @classmethod
    def process_pipeline(
        cls,
        image: Image.Image,
        grayscale: bool = True,
        deskew: bool = False,
        denoise: bool = False,
        contrast: bool = False,
        binarize: bool = False,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Run configured preprocessing pipeline on an image copy.
        Returns: (processed_image, applied_metadata)
        """
        # Create explicit copy to ensure original is never mutated
        processed = image.copy()
        meta: dict[str, Any] = {
            "original_size": image.size,
            "original_mode": image.mode,
            "steps_applied": [],
        }

        if deskew:
            processed, angle = cls.deskew_image(processed)
            meta["steps_applied"].append(f"deskew (angle={angle:.1f}deg)")
            meta["skew_angle"] = angle

        if grayscale and processed.mode != "L":
            processed = processed.convert("L")
            meta["steps_applied"].append("grayscale")

        if contrast:
            processed = cls.enhance_contrast(processed, factor=1.3)
            meta["steps_applied"].append("contrast_enhancement")

        if denoise:
            processed = cls.denoise_image(processed)
            meta["steps_applied"].append("median_denoise")

        if binarize:
            if processed.mode != "L":
                processed = processed.convert("L")
            processed = cls.binarize_otsu(processed)
            meta["steps_applied"].append("otsu_binarization")

        meta["final_size"] = processed.size
        meta["final_mode"] = processed.mode
        return processed, meta
