"""Security utilities for safe path traversal prevention and file sanitization."""

from pathlib import Path

from app.core.config import settings
from app.core.exceptions import PathTraversalError


def resolve_safe_path(base_dir: Path, user_path: str) -> Path:
    """
    Safely resolve a relative user/internal path within an approved base directory.
    Prevents path traversal attacks (e.g. '../', absolute path escapes, symlink traversal).
    """
    stripped = user_path.strip()
    # Reject leading slashes and Windows drive letters as absolute path attempts
    if stripped.startswith("/") or stripped.startswith("\\") or Path(stripped).is_absolute():
        raise PathTraversalError(user_path)

    base_resolved = base_dir.resolve()
    target_path = (base_resolved / stripped).resolve()

    # Verify that target_path is strictly within base_resolved
    try:
        target_path.relative_to(base_resolved)
    except ValueError:
        raise PathTraversalError(user_path)

    return target_path


def get_safe_dataset_path(relative_manifest_or_image: str) -> Path:
    """Resolve a path relative to the configured DATASET_ROOT."""
    return resolve_safe_path(settings.DATASET_ROOT, relative_manifest_or_image)


def get_safe_model_path(model_filename: str) -> Path:
    """Resolve a model file path relative to MODEL_DIR."""
    return resolve_safe_path(settings.MODEL_DIR, model_filename)
