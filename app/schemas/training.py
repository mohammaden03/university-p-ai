"""Training configuration and job tracking schemas."""

from typing import Any

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for controlled HTR model training run."""
    model_name: str = Field(default="persian-arabic-crnn-v2", description="New model identifier")
    train_manifest: str = Field(..., description="Training manifest path relative to DATASET_ROOT")
    val_manifest: str = Field(..., description="Validation manifest path relative to DATASET_ROOT")
    test_manifest: str | None = Field(None, description="Held-out test manifest path")
    epochs: int = Field(default=5, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=8, ge=1, le=64, description="Mini-batch size")
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1, description="Adam optimizer learning rate")
    image_height: int = Field(default=64, description="Standardized text line input height")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    resume_checkpoint: str | None = Field(None, description="Path to previous checkpoint to resume from")


class TrainingJobStatus(BaseModel):
    """Job status tracking for background training."""
    job_id: str
    status: str = Field(..., description="PENDING, RUNNING, COMPLETED, FAILED")
    model_name: str
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    val_cer: float | None = None
    checkpoint_path: str | None = None
    error_message: str | None = None
    started_at: str
    finished_at: str | None = None
    metrics_history: list[dict[str, Any]] = Field(default_factory=list)
