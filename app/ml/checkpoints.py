"""
Checkpoint management for saving, loading, and inspecting HTR models.
Includes model weights, vocabulary mapping, training history, and evaluation scores.
"""

import json
from pathlib import Path
from typing import Any

import torch

from app.core.logging import logger
from app.ml.model_arch import CRNN
from app.ml.persian_arabic import DEFAULT_VOCABULARY


def save_checkpoint(
    model: CRNN,
    checkpoint_path: Path,
    vocabulary: list,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """
    Save complete model checkpoint and companion JSON metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "vocabulary": vocabulary,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }

    # Save PyTorch binary
    torch.save(payload, checkpoint_path)

    # Save human-readable JSON metadata alongside
    meta_path = checkpoint_path.with_suffix(".json")
    meta_data = {
        "checkpoint_file": checkpoint_path.name,
        "vocabulary_size": len(vocabulary),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved checkpoint to {checkpoint_path} with metadata at {meta_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu"
) -> tuple[CRNN, list, dict[str, int], dict[int, str], dict[str, Any]]:
    """
    Load a model checkpoint and reconstruct the CRNN model and vocabulary.
    Returns: (model, vocabulary, char_to_idx, idx_to_char, metadata)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Safe PyTorch load with weights_only=False for complete state dict and vocab structure
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocabulary = data.get("vocabulary", DEFAULT_VOCABULARY)
    char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
    idx_to_char = {idx: char for idx, char in enumerate(vocabulary)}

    config = data.get("config", {})
    img_height = config.get("img_height", 64)
    rnn_hidden_size = config.get("rnn_hidden_size", 256)
    rnn_num_layers = config.get("rnn_num_layers", 2)

    model = CRNN(
        img_height=img_height,
        in_channels=1,
        num_classes=len(vocabulary),
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
    )

    model.load_state_dict(data["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "epoch": data.get("epoch", 0),
        "metrics": data.get("metrics", {}),
        "config": config,
    }

    return model, vocabulary, char_to_idx, idx_to_char, meta
