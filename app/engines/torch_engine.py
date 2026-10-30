"""
PyTorch CRNN Engine for Persian and Arabic manuscript HTR.
Integrates deep-learning text-line recognition with CTC decoding.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.core.exceptions import ModelLoadFailedError, ModelNotConfiguredError
from app.core.logging import logger
from app.engines.base import BaseEngine
from app.ml.checkpoints import load_checkpoint
from app.ml.model_arch import CRNN
from app.ml.persian_arabic import (
    CHAR_TO_IDX,
    DEFAULT_VOCABULARY,
    IDX_TO_CHAR,
    ctc_greedy_decode,
)
from app.schemas.models import ModelInfo, ModelStatus
from app.schemas.recognition import BoundingBox, RecognizedLine


class TorchCRNNEngine(BaseEngine):
    """
    Native PyTorch CRNN Engine supporting Persian and Arabic historical manuscripts.
    """

    def __init__(
        self,
        model_id: str = "persian-arabic-crnn-v1",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.status = ModelStatus.MODEL_NOT_CONFIGURED
        self.version = "1.0.0"
        self.model: CRNN | None = None
        self.vocabulary: list[str] = DEFAULT_VOCABULARY
        self.char_to_idx: dict[str, int] = CHAR_TO_IDX
        self.idx_to_char: dict[int, str] = IDX_TO_CHAR
        self.checkpoint_path: Path | None = None
        self.metadata: dict[str, Any] = {}

    def load_model(self, model_id: str, checkpoint_path: Path | None = None) -> None:
        """Load model weights and verify execution."""
        self.model_id = model_id
        self.status = ModelStatus.MODEL_LOADING

        if not checkpoint_path or not Path(checkpoint_path).exists():
            self.status = ModelStatus.MODEL_NOT_CONFIGURED
            logger.info(
                f"TorchCRNNEngine [{model_id}]: No checkpoint provided or file does not exist at {checkpoint_path}. "
                "State is MODEL_NOT_CONFIGURED."
            )
            return

        try:
            self.checkpoint_path = Path(checkpoint_path)
            model, vocab, c2i, i2c, meta = load_checkpoint(
                self.checkpoint_path,
                device=self.device
            )
            self.model = model
            self.vocabulary = vocab
            self.char_to_idx = c2i
            self.idx_to_char = i2c
            self.metadata = meta
            self.version = meta.get("config", {}).get("version", "1.0.0")

            # Verification forward pass with synthetic blank line
            dummy_input = torch.zeros((1, 1, self.model.img_height, 128), device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)

            self.status = ModelStatus.MODEL_READY
            logger.info(f"TorchCRNNEngine [{model_id}]: Successfully verified and loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            self.status = ModelStatus.MODEL_LOAD_FAILED
            logger.error(f"TorchCRNNEngine [{model_id}] failed to load: {e}")
            raise ModelLoadFailedError(model_id, str(e))

    def is_ready(self) -> bool:
        return self.status == ModelStatus.MODEL_READY and self.model is not None

    def get_status(self) -> ModelStatus:
        return self.status

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.model_id,
            name="Persian-Arabic Deep CRNN HTR Model",
            version=self.version,
            supported_languages=["PERSIAN", "ARABIC", "MIXED"],
            engine_type="torch_crnn",
            status=self.status,
            capabilities=[
                "text_line_recognition",
                "projection_profile_segmentation",
                "ctc_greedy_decoding",
                "confidence_scoring",
                "bidirectional_lstm_context",
                "harakat_diacritics_support",
            ],
            source="University Research Manuscript HTR Project",
            license="Academic & Research Use",
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            known_limitations=(
                "Trained primarily on Naskh, Thuluth, and Nasta'liq scripts. "
                "Extremely ornate calligraphic Shikasteh or heavily faded ink requires fine-tuning."
            ),
            evaluation_status=(
                f"Validation CER: {self.metadata.get('metrics', {}).get('val_cer', 'untested')}"
                if self.metadata else "Unchecked"
            ),
        )

    def _segment_lines(self, gray_img: Image.Image) -> list[tuple[Image.Image, BoundingBox]]:
        """
        Segment full page into horizontal text lines using projection profiling.
        Returns list of (line_crop_image, bounding_box).
        """
        w, h = gray_img.size
        # If image already looks like a single line (width >> height), return as is
        if w > h * 2.5 and h < 200:
            return [(gray_img, BoundingBox(x=0, y=0, w=w, h=h))]

        arr = np.array(gray_img)
        # Compute horizontal projection profile (sum of dark ink across rows)
        ink_density = np.sum(255 - arr, axis=1)
        mean_density = np.mean(ink_density)

        lines: list[tuple[Image.Image, BoundingBox]] = []
        in_line = False
        start_y = 0

        # Scan for line bands
        for y, val in enumerate(ink_density):
            if val > mean_density * 0.2 and not in_line:
                in_line = True
                start_y = max(0, y - 4)
            elif val <= mean_density * 0.2 and in_line:
                in_line = False
                end_y = min(h, y + 4)
                line_h = end_y - start_y
                if line_h >= 16:  # Ignore tiny specks
                    crop = gray_img.crop((0, start_y, w, end_y))
                    lines.append((crop, BoundingBox(x=0, y=start_y, w=w, h=line_h)))

        # Fallback if projection didn't find multiple distinct valleys
        if not lines:
            lines.append((gray_img, BoundingBox(x=0, y=0, w=w, h=h)))

        return lines

    def recognize(
        self,
        image: Image.Image,
        language: str = "PERSIAN",
        options: dict[str, Any] | None = None,
    ) -> tuple[str, float | None, list[RecognizedLine], list[str]]:
        if not self.is_ready():
            raise ModelNotConfiguredError(
                self.model_id,
                f"Model '{self.model_id}' is not in MODEL_READY state (current: {self.status.value}). "
                "Ensure trained weights are placed in the models directory."
            )

        options = options or {}
        warnings: list[str] = []

        # Convert to grayscale
        gray = image.convert("L")
        segments = self._segment_lines(gray)

        recognized_lines: list[RecognizedLine] = []
        line_texts: list[str] = []
        line_confidences: list[float] = []

        img_height = self.model.img_height if self.model else 64

        for idx, (line_img, bbox) in enumerate(segments, start=1):
            lw, lh = line_img.size
            if lh <= 0 or lw <= 0:
                continue

            target_w = max(int(lw * (img_height / float(lh))), 16)
            resized = line_img.resize((target_w, img_height), Image.Resampling.BILINEAR)

            arr = np.array(resized, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)  # [Time, 1, Classes]
                # Logits shape: [Time, Classes]
                step_logits = logits.squeeze(1).cpu().numpy().tolist()

            text, conf = ctc_greedy_decode(step_logits, self.idx_to_char)
            text = text.strip()

            recognized_lines.append(
                RecognizedLine(
                    line_id=idx,
                    text=text,
                    confidence=round(conf, 4) if conf > 0 else None,
                    bbox=bbox,
                    reading_order=idx,
                )
            )
            line_texts.append(text)
            if conf > 0:
                line_confidences.append(conf)

        full_text = "\n".join(t for t in line_texts if t)
        avg_confidence = (
            round(sum(line_confidences) / len(line_confidences), 4)
            if line_confidences else None
        )

        return full_text, avg_confidence, recognized_lines, warnings

    def release(self) -> None:
        """Release resources and clear PyTorch cache."""
        self.model = None
        self.status = ModelStatus.MODEL_NOT_CONFIGURED
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"TorchCRNNEngine [{self.model_id}]: Resources released.")
