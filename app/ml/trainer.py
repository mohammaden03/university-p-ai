"""
HTR Training pipeline with PyTorch CTC Loss, validation monitoring, and checkpoint saving.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.core.logging import logger
from app.ml.checkpoints import save_checkpoint
from app.ml.metrics import calculate_cer
from app.ml.model_arch import CRNN
from app.ml.persian_arabic import DEFAULT_VOCABULARY, ctc_greedy_decode


class HTRTrainer:
    """
    Manages model training, evaluation on validation split, and checkpoint preservation.
    """

    def __init__(
        self,
        model: CRNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        learning_rate: float = 0.001,
        device: str = "cpu",
        vocabulary: list | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.vocabulary = vocabulary or DEFAULT_VOCABULARY
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_epoch(self) -> float:
        """Run single training epoch and return average batch loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: [Time, Batch, NumClasses]
            logits = self.model(images)
            log_probs = logits.log_softmax(2)

            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("Encountered NaN or Inf loss in batch, skipping gradient update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        """
        Evaluate model on validation split.
        Returns: (average_loss, average_cer)
        """
        self.model.eval()
        total_loss = 0.0
        total_cer = 0.0
        sample_count = 0
        num_batches = 0

        for batch in dataloader:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            raw_texts = batch["texts"]

            logits = self.model(images)
            log_probs = logits.log_softmax(2)

            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

            # Transpose logits to [Batch, Time, Classes] for decoding
            b_logits = logits.permute(1, 0, 2).cpu().numpy()

            for i, step_logits in enumerate(b_logits):
                pred_text, _ = ctc_greedy_decode(step_logits.tolist(), self.idx_to_char)
                gt_text = raw_texts[i]
                cer, _, _, _, _ = calculate_cer(gt_text, pred_text)
                total_cer += cer
                sample_count += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_cer = total_cer / max(sample_count, 1)
        return avg_loss, avg_cer

    def run(
        self,
        epochs: int,
        model_name: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        """
        Execute training loop across specified epochs.
        Saves checkpoints and reports progress.
        """
        best_cer = float("inf")
        best_checkpoint_path = None

        for epoch in range(1, epochs + 1):
            start_t = time.time()
            train_loss = self.train_epoch()
            val_loss, val_cer = self.evaluate(self.val_loader)
            duration = time.time() - start_t

            logger.info(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f} ({duration:.2f}s)"
            )

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cer": val_cer,
                "epoch_duration_s": duration,
            }

            # Save checkpoint for this epoch
            ckpt_path = self.checkpoint_dir / f"{model_name}_epoch_{epoch}.pt"
            saved_path = save_checkpoint(
                model=self.model,
                checkpoint_path=ckpt_path,
                vocabulary=self.vocabulary,
                epoch=epoch,
                metrics=metrics,
                config={"img_height": self.model.img_height},
            )

            if val_cer < best_cer:
                best_cer = val_cer
                best_path = self.checkpoint_dir / f"{model_name}_best.pt"
                save_checkpoint(
                    model=self.model,
                    checkpoint_path=best_path,
                    vocabulary=self.vocabulary,
                    epoch=epoch,
                    metrics=metrics,
                    config={"img_height": self.model.img_height},
                )
                best_checkpoint_path = best_path

            if progress_callback:
                progress_callback({
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_cer": val_cer,
                    "checkpoint_path": str(saved_path),
                })

        return best_checkpoint_path or saved_path
