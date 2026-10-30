"""
Recognition endpoint for manuscript images.
Supports multipart/form-data file uploads and internal image references with NestJS backend.
"""


from fastapi import APIRouter, File, Form, Header, Request, UploadFile

from app.core.exceptions import InvalidImageError
from app.schemas.recognition import LanguageEnum, RecognitionResponse
from app.services.recognition_service import RecognitionService

router = APIRouter(prefix="/v1", tags=["Recognition"])


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_manuscript(
    request: Request,
    file: UploadFile | None = File(None, description="Manuscript image file (JPEG, PNG, TIFF, WebP)"),
    image_ref: str | None = Form(None, description="Internal relative image path within approved dataset root"),
    language: LanguageEnum = Form(LanguageEnum.PERSIAN, description="PERSIAN, ARABIC, or MIXED"),
    model_id: str | None = Form(None, description="Specific model ID to use (defaults to default model)"),
    deskew: bool = Form(False, description="Apply projection-profile deskewing"),
    denoise: bool = Form(False, description="Apply median denoising filter"),
    contrast: bool = Form(False, description="Enhance calligraphy ink contrast"),
    binarize: bool = Form(False, description="Apply Otsu binarization"),
    layout_analysis: bool = Form(True, description="Perform text-line segmentation and bounding box extraction"),
    x_request_id: str | None = Header(None, alias="X-Request-ID"),
):
    """
    Recognize handwritten Persian or Arabic text from a manuscript page or line image.
    Accepts either an uploaded multipart file or an internal approved image reference.
    """
    if file is None and (image_ref is None or not image_ref.strip()):
        raise InvalidImageError("Either a multipart 'file' upload or an 'image_ref' form field must be provided.")

    image_bytes = None
    filename = "manuscript.jpg"

    if file is not None:
        filename = file.filename or "upload.jpg"
        image_bytes = await file.read()

    response = RecognitionService.process_recognition(
        image_bytes=image_bytes,
        image_ref=image_ref,
        language=language,
        model_id=model_id,
        deskew=deskew,
        denoise=denoise,
        contrast=contrast,
        binarize=binarize,
        layout_analysis=layout_analysis,
        filename=filename,
        request_id=x_request_id,
    )

    return response
