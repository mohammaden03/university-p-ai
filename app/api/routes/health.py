"""
Health and readiness endpoints.
Strictly separates HTTP API health from machine learning model readiness.
"""

from datetime import UTC, datetime

import torch
from fastapi import APIRouter

from app.core.config import settings
from app.schemas.health import HealthResponse, ReadinessResponse
from app.schemas.models import ModelStatus
from app.services.model_registry import registry

router = APIRouter(tags=["Health & Readiness"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """
    Liveness probe.
    Confirms HTTP server process is running and responding.
    Does NOT imply that an HTR model is loaded or ready.
    """
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        ready=True,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def get_readiness():
    """
    Readiness probe.
    Reports separate dependency checks and active recognition model readiness.
    """
    default_engine = registry.get_engine(registry.default_model_id)
    default_status = default_engine.get_status()
    model_ready = (default_status == ModelStatus.MODEL_READY)

    dependencies = {
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": settings.INFERENCE_DEVICE,
        },
        "storage": {
            "model_dir_exists": settings.MODEL_DIR.exists(),
            "dataset_root_exists": settings.DATASET_ROOT.exists(),
            "checkpoint_dir_exists": settings.CHECKPOINT_DIR.exists(),
        },
        "default_model": {
            "id": registry.default_model_id,
            "status": default_status.value,
            "ready": model_ready,
        },
    }

    return ReadinessResponse(
        ready=model_ready,
        service_status="ready",
        default_model_status=default_status.value,
        dependencies=dependencies,
        timestamp=datetime.now(UTC).isoformat(),
    )
