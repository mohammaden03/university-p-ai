"""
Asynchronous training service orchestrating reproducible HTR model training.
Runs in a background thread to prevent blocking the FastAPI event loop.
"""

import concurrent.futures
import uuid
from datetime import UTC, datetime
from typing import Any

import torch
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.exceptions import TrainingJobError
from app.core.logging import logger
from app.ml.dataset import HTRLineDataset, ctc_collate_fn
from app.ml.model_arch import CRNN
from app.ml.persian_arabic import CHAR_TO_IDX, DEFAULT_VOCABULARY
from app.ml.trainer import HTRTrainer
from app.schemas.training import TrainingConfig, TrainingJobStatus
from app.services.dataset_service import DatasetService
from app.services.model_registry import registry


class TrainingService:
    """Manages asynchronous training jobs, status queries, and checkpoint registration."""

    def __init__(self):
        self._jobs: dict[str, TrainingJobStatus] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def get_job(self, job_id: str) -> TrainingJobStatus:
        if job_id not in self._jobs:
            raise TrainingJobError(f"Training job '{job_id}' not found.", status_code=404)
        return self._jobs[job_id]

    def list_jobs(self) -> list[TrainingJobStatus]:
        return list(self._jobs.values())

    def start_training(self, config: TrainingConfig) -> TrainingJobStatus:
        # Validate train and validation manifests first
        DatasetService.validate_dataset(config.train_manifest, verify_images=True)
        DatasetService.validate_dataset(config.val_manifest, verify_images=True)

        job_id = f"job-{uuid.uuid4().hex[:8]}"
        now_str = datetime.now(UTC).isoformat()

        job_status = TrainingJobStatus(
            job_id=job_id,
            status="PENDING",
            model_name=config.model_name,
            current_epoch=0,
            total_epochs=config.epochs,
            started_at=now_str,
        )
        self._jobs[job_id] = job_status

        # Dispatch execution to background thread pool
        self._executor.submit(self._run_training_job, job_id, config)
        return job_status

    def _run_training_job(self, job_id: str, config: TrainingConfig) -> None:
        job = self._jobs[job_id]
        job.status = "RUNNING"
        logger.info(f"Started training job [{job_id}] for model '{config.model_name}'")

        try:
            # Set random seeds for reproducibility
            torch.manual_seed(config.random_seed)

            # Load datasets
            train_samples = DatasetService.load_manifest(config.train_manifest)
            val_samples = DatasetService.load_manifest(config.val_manifest)

            train_dataset = HTRLineDataset(
                samples=train_samples,
                dataset_root=settings.DATASET_ROOT,
                img_height=config.image_height,
                char_to_idx=CHAR_TO_IDX,
            )
            val_dataset = HTRLineDataset(
                samples=val_samples,
                dataset_root=settings.DATASET_ROOT,
                img_height=config.image_height,
                char_to_idx=CHAR_TO_IDX,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=ctc_collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=ctc_collate_fn,
            )

            # Initialize model
            device = settings.INFERENCE_DEVICE
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            model = CRNN(
                img_height=config.image_height,
                in_channels=1,
                num_classes=len(DEFAULT_VOCABULARY),
                rnn_hidden_size=256,
                rnn_num_layers=2,
            )

            # If resuming
            if config.resume_checkpoint:
                from app.ml.checkpoints import load_checkpoint
                ckpt_p = settings.BASE_DIR / config.resume_checkpoint
                if ckpt_p.exists():
                    loaded_m, _, _, _, _ = load_checkpoint(ckpt_p, device=device)
                    model.load_state_dict(loaded_m.state_dict())
                    logger.info(f"Resumed weights from {ckpt_p}")

            def progress_callback(update: dict[str, Any]):
                job.current_epoch = update["epoch"]
                job.train_loss = round(update["train_loss"], 4)
                job.val_loss = round(update["val_loss"], 4)
                job.val_cer = round(update["val_cer"], 4)
                job.checkpoint_path = update["checkpoint_path"]
                job.metrics_history.append(update)

            trainer = HTRTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=settings.CHECKPOINT_DIR,
                learning_rate=config.learning_rate,
                device=device,
                vocabulary=DEFAULT_VOCABULARY,
            )

            best_checkpoint = trainer.run(
                epochs=config.epochs,
                model_name=config.model_name,
                progress_callback=progress_callback,
            )

            job.status = "COMPLETED"
            job.checkpoint_path = str(best_checkpoint)
            job.finished_at = datetime.now(UTC).isoformat()
            logger.info(f"Training job [{job_id}] completed successfully. Saved: {best_checkpoint}")

            # Reload model into registry so it is immediately usable for inference
            try:
                registry.reload_checkpoint("persian-arabic-crnn-v1", best_checkpoint)
            except Exception as e:
                logger.warning(f"Could not automatically update active registry engine: {e}")

        except Exception as e:
            job.status = "FAILED"
            job.error_message = str(e)
            job.finished_at = datetime.now(UTC).isoformat()
            logger.error(f"Training job [{job_id}] failed: {e}", exc_info=True)


training_service = TrainingService()
