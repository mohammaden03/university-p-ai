"""
Training endpoints.
Provides asynchronous job dispatch, progress polling, and checkpoint registration.
"""


from fastapi import APIRouter

from app.schemas.training import TrainingConfig, TrainingJobStatus
from app.services.training_service import training_service

router = APIRouter(prefix="/v1/train", tags=["Training"])


@router.post("", response_model=TrainingJobStatus, status_code=202)
async def start_training_run(config: TrainingConfig):
    """
    Initiate controlled background model training run on validated dataset manifests.
    Returns 202 Accepted with a tracking job ID. Does not block the API event loop.
    """
    return training_service.start_training(config)


@router.get("/{job_id}", response_model=TrainingJobStatus)
async def get_training_job_status(job_id: str):
    """
    Retrieve live training progress, current epoch, loss, and checkpoint paths.
    """
    return training_service.get_job(job_id)


@router.get("", response_model=list[TrainingJobStatus])
async def list_training_jobs():
    """
    List all training jobs in current session.
    """
    return training_service.list_jobs()
