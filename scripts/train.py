"""
Command-line HTR training script for Persian and Arabic manuscript recognition.
Usage:
  python3 scripts/train.py --train-manifest sample_persian_arabic/train.jsonl --val-manifest sample_persian_arabic/val.jsonl --epochs 3
"""

import argparse
import sys
from pathlib import Path

# Ensure app package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from app.core.config import settings
from app.ml.dataset import HTRLineDataset, ctc_collate_fn
from app.ml.model_arch import CRNN
from app.ml.persian_arabic import DEFAULT_VOCABULARY
from app.ml.trainer import HTRTrainer
from app.services.dataset_service import DatasetService


def main():
    parser = argparse.ArgumentParser(description="Train Persian/Arabic Manuscript CRNN HTR Model")
    parser.add_argument("--model-name", type=str, default="persian-arabic-crnn-v1", help="Target model identifier")
    parser.add_argument("--train-manifest", type=str, default="sample_persian_arabic/train.jsonl", help="Train manifest path")
    parser.add_argument("--val-manifest", type=str, default="sample_persian_arabic/val.jsonl", help="Validation manifest path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img-height", type=int, default=64, help="Normalized text line image height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print("==================================================================")
    print(" Persian & Arabic Manuscript HTR - Model Training Pipeline")
    print("==================================================================")
    print(f"Model ID: {args.model_name}")
    print(f"Train Manifest: {args.train_manifest}")
    print(f"Val Manifest:   {args.val_manifest}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Device: {args.device} | Seed: {args.seed}")

    torch.manual_seed(args.seed)

    # 1. Validate manifests
    print("\nAuditing dataset manifests...")
    DatasetService.validate_dataset(args.train_manifest)
    DatasetService.validate_dataset(args.val_manifest)
    print("Manifest validation passed!")

    # 2. Build datasets
    train_samples = DatasetService.load_manifest(args.train_manifest)
    val_samples = DatasetService.load_manifest(args.val_manifest)

    train_ds = HTRLineDataset(train_samples, settings.DATASET_ROOT, img_height=args.img_height)
    val_ds = HTRLineDataset(val_samples, settings.DATASET_ROOT, img_height=args.img_height)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn)

    # 3. Model
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = CRNN(
        img_height=args.img_height,
        in_channels=1,
        num_classes=len(DEFAULT_VOCABULARY),
        rnn_hidden_size=256,
        rnn_num_layers=2,
    )

    trainer = HTRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=settings.CHECKPOINT_DIR,
        learning_rate=args.lr,
        device=device,
        vocabulary=DEFAULT_VOCABULARY,
    )

    print("\nStarting training loop...")
    saved_checkpoint = trainer.run(
        epochs=args.epochs,
        model_name=args.model_name,
    )

    print("\nTraining completed successfully!")
    print(f"Best checkpoint saved at: {saved_checkpoint}")


if __name__ == "__main__":
    main()
