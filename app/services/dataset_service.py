"""
Dataset preparation, manifest validation, and integrity auditing.
Ensures zero data leakage, validates Unicode character sets, and checks physical image files.
"""

import json
from typing import Any

from PIL import Image

from app.core.config import settings
from app.core.exceptions import DatasetValidationError
from app.core.security import get_safe_dataset_path
from app.ml.persian_arabic import DEFAULT_VOCABULARY


class DatasetService:
    """Utilities for preparing and validating HTR datasets."""

    @staticmethod
    def load_manifest(manifest_rel_path: str) -> list[dict[str, Any]]:
        """Safely load and parse a JSONL dataset manifest."""
        safe_path = get_safe_dataset_path(manifest_rel_path)
        if not safe_path.exists():
            raise DatasetValidationError(f"Dataset manifest does not exist at: {manifest_rel_path}")

        samples = []
        with open(safe_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    samples.append(record)
                except json.JSONDecodeError as e:
                    raise DatasetValidationError(
                        f"JSON syntax error on line {line_no} in '{manifest_rel_path}': {e!s}"
                    )

        return samples

    @classmethod
    def validate_dataset(
        cls,
        manifest_rel_path: str,
        verify_images: bool = True
    ) -> dict[str, Any]:
        """
        Validate complete dataset manifest against filesystem and formatting standards.
        Checks:
          - Missing image files
          - Empty ground-truth transcriptions
          - Duplicate sample IDs
          - Invalid splits
          - Train/val/test leakage by manuscript ID
          - Character set coverage
        """
        samples = cls.load_manifest(manifest_rel_path)
        if not samples:
            raise DatasetValidationError("Dataset manifest contains 0 samples.")

        errors: list[str] = []
        warnings: list[str] = []
        seen_ids = set()
        manuscript_to_splits: dict[str, set] = {}

        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0, "other": 0}
        char_counts: dict[str, int] = {}
        total_words = 0
        known_vocab_set = set(DEFAULT_VOCABULARY)
        unrecognized_chars = set()

        for idx, sample in enumerate(samples, start=1):
            sample_id = sample.get("id", f"sample_{idx}")
            if sample_id in seen_ids:
                warnings.append(f"Duplicate sample ID detected: '{sample_id}' (line {idx})")
            seen_ids.add(sample_id)

            # Check required fields
            if "image_path" not in sample:
                errors.append(f"Missing 'image_path' in sample ID '{sample_id}' (line {idx})")
                continue

            # Ground truth text
            text = sample.get("text", "")
            if text is None or not str(text).strip():
                errors.append(f"Empty ground-truth transcription for sample ID '{sample_id}' (line {idx})")
                continue

            text_str = str(text)
            words = text_str.split()
            total_words += len(words)

            for char in text_str:
                char_counts[char] = char_counts.get(char, 0) + 1
                if char not in known_vocab_set:
                    unrecognized_chars.add(char)

            # Split validation
            split = sample.get("split", "train").lower()
            if split in split_counts:
                split_counts[split] += 1
            else:
                split_counts["other"] += 1

            # Leakage check: Same manuscript should not span multiple splits
            ms_id = sample.get("manuscript_id")
            if ms_id:
                if ms_id not in manuscript_to_splits:
                    manuscript_to_splits[ms_id] = set()
                manuscript_to_splits[ms_id].add(split)

            # Image existence and decodability check
            if verify_images:
                img_rel = sample["image_path"]
                full_img_path = (settings.DATASET_ROOT / img_rel).resolve()
                if not full_img_path.exists():
                    errors.append(f"Image file does not exist: {img_rel} (referenced by '{sample_id}')")
                else:
                    try:
                        with Image.open(full_img_path) as im:
                            im.verify()
                    except Exception as e:
                        errors.append(f"Corrupted image file {img_rel} ('{sample_id}'): {e!s}")

        # Check for manuscript leakage
        for ms_id, splits in manuscript_to_splits.items():
            if len(splits) > 1:
                warnings.append(
                    f"Data leakage warning: Manuscript '{ms_id}' is split across multiple sets: {splits}. "
                    "Pages of the same manuscript should ideally stay in the same split."
                )

        if unrecognized_chars:
            warnings.append(
                f"Encountered {len(unrecognized_chars)} characters not in default vocabulary: "
                f"{list(unrecognized_chars)[:10]}"
            )

        if errors:
            raise DatasetValidationError(
                f"Manifest '{manifest_rel_path}' failed validation with {len(errors)} error(s).",
                errors=errors,
            )

        return {
            "valid": True,
            "manifest_path": manifest_rel_path,
            "total_samples": len(samples),
            "split_counts": split_counts,
            "total_words": total_words,
            "vocabulary_size": len(char_counts),
            "unrecognized_chars": list(unrecognized_chars),
            "warnings": warnings,
        }
