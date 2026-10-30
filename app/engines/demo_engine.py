"""
Deterministic Demo / Test Engine for automated testing and sandbox environments.
Permits end-to-end verification of preprocessing, API contracts, layout extraction,
and response serialization without requiring large pre-downloaded model weights.
"""

from pathlib import Path
from typing import Any

from PIL import Image

from app.engines.base import BaseEngine
from app.schemas.models import ModelInfo, ModelStatus
from app.schemas.recognition import BoundingBox, RecognizedLine


class DemoEngine(BaseEngine):
    """
    Deterministic verification engine for automated testing and sandbox evaluation.
    """

    def __init__(
        self,
        model_id: str = "demo-persian-naskh-v1",
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = device
        self.status = ModelStatus.MODEL_READY
        self.version = "1.0.0-test"

    def load_model(self, model_id: str, checkpoint_path: Path | None = None) -> None:
        self.model_id = model_id
        self.status = ModelStatus.MODEL_READY

    def is_ready(self) -> bool:
        return self.status == ModelStatus.MODEL_READY

    def get_status(self) -> ModelStatus:
        return self.status

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.model_id,
            name="Deterministic Persian/Arabic Test Engine",
            version=self.version,
            supported_languages=["PERSIAN", "ARABIC", "MIXED"],
            engine_type="demo",
            status=self.status,
            capabilities=[
                "deterministic_test_inference",
                "projection_line_segmentation",
                "confidence_calculation",
            ],
            source="Automated Test Suite",
            license="MIT",
            checkpoint_path=None,
            known_limitations="For API contract and pipeline testing only.",
            evaluation_status="Deterministic benchmark passed",
        )

    def recognize(
        self,
        image: Image.Image,
        language: str = "PERSIAN",
        options: dict[str, Any] | None = None,
    ) -> tuple[str, float | None, list[RecognizedLine], list[str]]:
        w, h = image.size

        # Simple deterministic rule based on language and image dimension
        if language == "ARABIC":
            line1 = "بسم الله الرحمن الرحيم"
            line2 = "الحمد لله رب العالمين"
        elif language == "MIXED":
            line1 = "قال الحکیم فی دیباجة الکتاب"
            line2 = "به نام خداوند جان و خرد"
        else:
            line1 = "به نام خداوند جان و خرد"
            line2 = "کزین برتر اندیشه برنگذرد"

        line_h = max(h // 2, 20)
        lines = [
            RecognizedLine(
                line_id=1,
                text=line1,
                confidence=0.9420,
                bbox=BoundingBox(x=0, y=0, w=w, h=line_h),
                reading_order=1,
            ),
            RecognizedLine(
                line_id=2,
                text=line2,
                confidence=0.9150,
                bbox=BoundingBox(x=0, y=line_h, w=w, h=h - line_h),
                reading_order=2,
            ),
        ]

        full_text = f"{line1}\n{line2}"
        avg_conf = 0.9285
        return full_text, avg_conf, lines, []

    def release(self) -> None:
        self.status = ModelStatus.MODEL_NOT_CONFIGURED
