"""
Recognition service coordinating preprocessing, model engine invocation,
and result formatting.
"""

import time
import uuid

from app.core.security import get_safe_dataset_path
from app.schemas.recognition import (
    LanguageEnum,
    RecognitionModelMeta,
    RecognitionResponse,
)
from app.services.model_registry import registry
from app.services.preprocessing_service import PreprocessingService


class RecognitionService:
    """Orchestrates image decoding, preprocessing, and HTR model inference."""

    @staticmethod
    def process_recognition(
        image_bytes: bytes | None = None,
        image_ref: str | None = None,
        language: LanguageEnum = LanguageEnum.PERSIAN,
        model_id: str | None = None,
        deskew: bool = False,
        denoise: bool = False,
        contrast: bool = False,
        binarize: bool = False,
        layout_analysis: bool = True,
        filename: str = "manuscript.jpg",
        request_id: str | None = None,
    ) -> RecognitionResponse:
        req_id = request_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        # Step 1: Obtain raw image bytes from upload or safe internal reference
        if image_bytes is None:
            if not image_ref:
                raise ValueError("Either an image file upload or internal image_ref must be provided.")
            safe_path = get_safe_dataset_path(image_ref)
            if not safe_path.exists():
                raise FileNotFoundError(f"Referenced manuscript image not found: {image_ref}")
            with open(safe_path, "rb") as f:
                image_bytes = f.read()
            filename = safe_path.name

        # Step 2: Decode and validate
        raw_img = PreprocessingService.decode_and_validate_image(image_bytes, filename=filename)

        # Step 3: Run requested preprocessing pipeline
        processed_img, prep_meta = PreprocessingService.process_pipeline(
            raw_img,
            grayscale=True,
            deskew=deskew,
            denoise=denoise,
            contrast=contrast,
            binarize=binarize,
        )

        # Step 4: Select engine
        target_model_id = model_id or registry.default_model_id
        engine = registry.get_engine(target_model_id)
        model_info = engine.get_model_info()

        # Step 5: Execute inference
        text, confidence, lines, engine_warnings = engine.recognize(
            image=processed_img,
            language=language.value,
            options={
                "layout_analysis": layout_analysis,
                "preprocessing": prep_meta,
            },
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        warnings = list(engine_warnings)
        if prep_meta.get("steps_applied"):
            warnings.append(f"Applied preprocessing: {', '.join(prep_meta['steps_applied'])}")

        return RecognitionResponse(
            status="success",
            request_id=req_id,
            text=text,
            language=language,
            model=RecognitionModelMeta(id=model_info.id, version=model_info.version),
            processing_time_ms=elapsed_ms,
            confidence=confidence,
            lines=lines,
            warnings=warnings,
        )
