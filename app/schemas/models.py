"""Model registry schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    """Model lifecycle state."""
    MODEL_READY = "MODEL_READY"
    MODEL_NOT_CONFIGURED = "MODEL_NOT_CONFIGURED"
    MODEL_LOADING = "MODEL_LOADING"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"


class ModelInfo(BaseModel):
    """Detailed model metadata and capabilities."""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    version: str = Field(..., description="Model version string")
    supported_languages: list[str] = Field(..., description="Supported languages, e.g. PERSIAN, ARABIC, MIXED")
    engine_type: str = Field(..., description="Engine adapter type: torch_crnn, kraken, demo")
    status: ModelStatus = Field(..., description="Current availability state")
    capabilities: list[str] = Field(..., description="List of supported engine features")
    source: str | None = Field(None, description="Model source or institution")
    license: str | None = Field(None, description="Model weights license if known")
    checkpoint_path: str | None = Field(None, description="Local path to model checkpoint")
    known_limitations: str | None = Field(None, description="Known script or calligraphy constraints")
    evaluation_status: str | None = Field(None, description="Summary of last known evaluation")


class ModelsListResponse(BaseModel):
    """List of all configured models in the registry."""
    default_model_id: str
    models: list[ModelInfo]
