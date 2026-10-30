"""Custom application exceptions with HTTP and domain mapping."""

from typing import Any

from fastapi import HTTPException, status


class HTRException(HTTPException):
    """Base class for all domain-specific HTR service exceptions."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "error_code": error_code,
                "message": message,
                "details": details or {},
            },
        )
        self.error_code = error_code
        self.message = message
        self.details = details or {}


class ModelNotConfiguredError(HTRException):
    """Raised when an inference is requested but no model is configured or ready."""

    def __init__(self, model_id: str, message: str | None = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_NOT_CONFIGURED",
            message=message or f"Recognition model '{model_id}' is not loaded or configured. Please train or deploy model weights first.",
            details={"model_id": model_id, "status": "MODEL_NOT_CONFIGURED"},
        )


class ModelLoadFailedError(HTRException):
    """Raised when loading model weights fails."""

    def __init__(self, model_id: str, reason: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_LOAD_FAILED",
            message=f"Failed to load weights for model '{model_id}': {reason}",
            details={"model_id": model_id, "reason": reason, "status": "MODEL_LOAD_FAILED"},
        )


class ModelNotFoundError(HTRException):
    """Raised when a requested model ID does not exist in registry."""

    def __init__(self, model_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="MODEL_NOT_FOUND",
            message=f"Model identifier '{model_id}' is not recognized in the registry.",
            details={"model_id": model_id},
        )


class InvalidImageError(HTRException):
    """Raised when uploaded or referenced image is invalid, unreadable, or corrupted."""

    def __init__(self, reason: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="INVALID_IMAGE",
            message=f"Image decoding failed or corrupted: {reason}",
            details={"reason": reason},
        )


class ImageTooLargeError(HTRException):
    """Raised when uploaded file exceeds allowed byte size or pixel resolution."""

    def __init__(self, current_size: int, max_size: int, dimension_type: str = "bytes"):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            error_code="IMAGE_TOO_LARGE",
            message=f"Image exceeds maximum permissible {dimension_type} limit: {current_size} > {max_size}",
            details={"current": current_size, "limit": max_size, "unit": dimension_type},
        )


class PathTraversalError(HTRException):
    """Raised when an internal file reference tries to escape approved root boundaries."""

    def __init__(self, requested_path: str):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="PATH_TRAVERSAL_DETECTED",
            message="Internal file reference violates security boundaries and cannot escape approved storage root.",
            details={"requested_path": requested_path},
        )


class DatasetValidationError(HTRException):
    """Raised when dataset manifest or samples fail schema, file existence, or label checks."""

    def __init__(self, reason: str, errors: list | None = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="DATASET_VALIDATION_ERROR",
            message=f"Dataset validation failed: {reason}",
            details={"errors": errors or []},
        )


class TrainingJobError(HTRException):
    """Raised when training configuration or run encounters an error."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(
            status_code=status_code,
            error_code="TRAINING_ERROR",
            message=message,
        )
