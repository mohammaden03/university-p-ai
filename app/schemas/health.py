"""Health and readiness schemas."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness check response."""
    status: str = Field(..., description="Service overall operational status ('ok', 'degraded')")
    version: str = Field(..., description="Service application semver version")
    ready: bool = Field(..., description="Whether API is initialized to accept HTTP requests")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class ReadinessResponse(BaseModel):
    """Readiness probe checking dependencies and model availability."""
    ready: bool = Field(..., description="Whether service and models are ready for inference")
    service_status: str = Field(..., description="HTTP server status")
    default_model_status: str = Field(..., description="MODEL_READY, MODEL_NOT_CONFIGURED, etc.")
    dependencies: dict[str, Any] = Field(..., description="Status breakdown of dependencies and storage")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
