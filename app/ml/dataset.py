"""
Dataset loading and batch collation for PyTorch HTR training and evaluation.
Handles JSONL manifests, dynamic line image aspect ratios, and CTC target encoding.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.ml.persian_arabic import CHAR_TO_IDX, text_to_indices


class HTRLineDataset(Dataset):
    """
    PyTorch Dataset representing Persian and Arabic manuscript text lines.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        dataset_root: Path,
        img_height: int = 64,
        char_to_idx: dict[str, int] = CHAR_TO_IDX,
    ):
        self.samples = samples
        self.dataset_root = Path(dataset_root)
        self.img_height = img_height
        self.char_to_idx = char_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        image_rel_path = item["image_path"]
        full_image_path = (self.dataset_root / image_rel_path).resolve()

        if not full_image_path.exists():
            raise FileNotFoundError(f"Manuscript image not found: {full_image_path}")

        # Open image and convert to grayscale
        with Image.open(full_image_path) as img:
            img = img.convert("L")

            # Check optional bounding box cropping for page-level manifests
            if item.get("bbox"):
                b = item["bbox"]
                box = (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
                img = img.crop(box)

            # Resize to target height while preserving aspect ratio
            w, h = img.size
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid image dimensions: {w}x{h}")

            new_w = max(int(w * (self.img_height / float(h))), 16)
            img = img.resize((new_w, self.img_height), Image.Resampling.BILINEAR)

            # Convert to normalized float32 tensor [1, H, W] in [-1, 1]
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

        text = item.get("text", "")
        target_indices = text_to_indices(text, self.char_to_idx)

        return {
            "image": tensor,
            "target": torch.tensor(target_indices, dtype=torch.long),
            "text": text,
            "sample_id": str(item.get("id", idx)),
            "width": new_w,
        }


def ctc_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for dynamic-width CTC batching.
    Pads all images in the batch to the maximum width in that batch.
    """
    max_w = max(b["width"] for b in batch)
    batch_size = len(batch)
    img_height = batch[0]["image"].shape[1]

    # Pre-allocate padded image batch [B, 1, H, max_W] with padding value (1.0 = white background)
    padded_images = torch.ones((batch_size, 1, img_height, max_w), dtype=torch.float32)

    target_list = []
    target_lengths = []
    input_lengths = []
    texts = []
    sample_ids = []

    for i, b in enumerate(batch):
        img = b["image"]
        w = b["width"]
        padded_images[i, :, :, :w] = img

        target = b["target"]
        target_list.append(target)
        target_lengths.append(len(target))

        # Model downsamples width by a factor of 4 across max-pooling layers
        seq_len = max(w // 4, 1)
        input_lengths.append(seq_len)

        texts.append(b["text"])
        sample_ids.append(b["sample_id"])

    # Concatenate 1D targets for PyTorch CTCLoss
    if target_list:
        targets = torch.cat(target_list)
    else:
        targets = torch.empty(0, dtype=torch.long)

    return {
        "images": padded_images,
        "targets": targets,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
        "sample_ids": sample_ids,
    }
