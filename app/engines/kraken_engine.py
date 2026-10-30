"""
Kraken HTR Engine Adapter.
Integrates Kraken's bidirectional text recognition, baseline segmentation,
and .mlmodel loading when kraken dependencies and models are configured.
"""

from pathlib import Path
from typing import Any

from PIL import Image

from app.core.exceptions import ModelNotConfiguredError
from app.core.logging import logger
from app.engines.base import BaseEngine
from app.schemas.models import ModelInfo, ModelStatus
from app.schemas.recognition import BoundingBox, RecognizedLine

# Check whether kraken is available in the current environment
try:
    from kraken import rpred
    from kraken.lib import models as kraken_models
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False


class KrakenEngine(BaseEngine):
    """
    Adapter for the Kraken open-source HTR engine for historical documents.
    """

    def __init__(
        self,
        model_id: str = "kraken-arabic-persian-v1",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = device
        self.status = ModelStatus.MODEL_NOT_CONFIGURED
        self.version = "1.0.0"
        self.model = None
        self.checkpoint_path: Path | None = None

    def load_model(self, model_id: str, checkpoint_path: Path | None = None) -> None:
        self.model_id = model_id

        if not KRAKEN_AVAILABLE:
            self.status = ModelStatus.MODEL_NOT_CONFIGURED
            logger.info(
                f"KrakenEngine [{model_id}]: 'kraken' package is not installed in current Python environment. "
                "State is MODEL_NOT_CONFIGURED."
            )
            return

        if not checkpoint_path or not Path(checkpoint_path).exists():
            self.status = ModelStatus.MODEL_NOT_CONFIGURED
            logger.info(
                f"KrakenEngine [{model_id}]: No .mlmodel checkpoint found at '{checkpoint_path}'. "
                "State is MODEL_NOT_CONFIGURED."
            )
            return

        try:
            self.checkpoint_path = Path(checkpoint_path)
            self.model = kraken_models.load_any(str(self.checkpoint_path), device=self.device)
            self.status = ModelStatus.MODEL_READY
            logger.info(f"KrakenEngine [{model_id}]: Successfully loaded Kraken model from {checkpoint_path}")
        except Exception as e:
            self.status = ModelStatus.MODEL_LOAD_FAILED
            logger.error(f"KrakenEngine [{model_id}] failed to load: {e}")

    def is_ready(self) -> bool:
        return self.status == ModelStatus.MODEL_READY and self.model is not None

    def get_status(self) -> ModelStatus:
        return self.status

    def get_model_info(self) -> ModelInfo:
        limitations = (
            "Kraken requires compiled C/C++ dependencies (libraqm, fribidi, harfbuzz). "
            "Pretrained generic Arabic models often suffer high CER on historical Persian calligraphic manuscripts "
            "(e.g. Nasta'liq sloping baselines and overlapping words). Custom baseline training is required."
        )
        if not KRAKEN_AVAILABLE:
            limitations = "Kraken ML library is not installed in the current environment. " + limitations

        return ModelInfo(
            id=self.model_id,
            name="Kraken Historical Document HTR Engine",
            version=self.version,
            supported_languages=["ARABIC", "PERSIAN"],
            engine_type="kraken",
            status=self.status,
            capabilities=[
                "kraken_mlmodel_inference",
                "baseline_segmentation",
                "bidi_reordering",
                "historical_document_layout",
            ],
            source="Kraken OCR/HTR Project (Open Source)",
            license="Apache 2.0",
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            known_limitations=limitations,
            evaluation_status="Pending verified Persian manuscript checkpoint",
        )

    def recognize(
        self,
        image: Image.Image,
        language: str = "PERSIAN",
        options: dict[str, Any] | None = None,
    ) -> tuple[str, float | None, list[RecognizedLine], list[str]]:
        if not self.is_ready():
            raise ModelNotConfiguredError(
                self.model_id,
                f"Kraken model '{self.model_id}' is not loaded. Current status: {self.status.value}. "
                "Please configure a valid Kraken .mlmodel file or use the torch_crnn engine."
            )

        # When kraken is installed and model loaded
        try:
            # Layout and line extraction using kraken
            records = list(rpred.rpred(network=self.model, im=image))
            recognized_lines: list[RecognizedLine] = []
            line_texts: list[str] = []
            confidences: list[float] = []

            for i, record in enumerate(records, start=1):
                text = record.prediction
                line_texts.append(text)
                # Compute confidence from character probabilities if available
                conf = None
                if hasattr(record, "probabilities") and record.probabilities:
                    conf = float(sum(record.probabilities) / len(record.probabilities))
                    confidences.append(conf)

                bbox = None
                if hasattr(record, "bounds"):
                    b = record.bounds
                    bbox = BoundingBox(x=b[0], y=b[1], w=b[2] - b[0], h=b[3] - b[1])

                recognized_lines.append(
                    RecognizedLine(
                        line_id=i,
                        text=text,
                        confidence=round(conf, 4) if conf else None,
                        bbox=bbox,
                        reading_order=i,
                    )
                )

            full_text = "\n".join(line_texts)
            avg_conf = (
                round(sum(confidences) / len(confidences), 4)
                if confidences else None
            )
            return full_text, avg_conf, recognized_lines, []
        except Exception as e:
            logger.error(f"Kraken recognition error: {e}")
            raise

    def release(self) -> None:
        self.model = None
        self.status = ModelStatus.MODEL_NOT_CONFIGURED
