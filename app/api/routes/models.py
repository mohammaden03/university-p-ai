"""
Models information endpoints.
Reports registered engines, availability states, supported languages, and capabilities.
"""

from fastapi import APIRouter

from app.schemas.models import ModelInfo, ModelsListResponse
from app.services.model_registry import registry

router = APIRouter(prefix="/v1/models", tags=["Models"])


@router.get("", response_model=ModelsListResponse)
async def list_models():
    """
    List all configured HTR models, their lifecycle status, and engine capabilities.
    """
    models = registry.list_models()
    return ModelsListResponse(
        default_model_id=registry.default_model_id,
        models=models,
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Retrieve metadata and current status for a specific model ID.
    """
    engine = registry.get_engine(model_id)
    return engine.get_model_info()
