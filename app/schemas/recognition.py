"""Recognition request and response schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class LanguageEnum(str, Enum):
    """Supported manuscript languages."""
    PERSIAN = "PERSIAN"
    ARABIC = "ARABIC"
    MIXED = "MIXED"


class BoundingBox(BaseModel):
    """
    Bounding box in standard image coordinates.
    Coordinate system:
      - Origin (0,0): Top-left corner of the source image.
      - x: Horizontal distance in pixels from the left edge.
      - y: Vertical distance in pixels from the top edge.
      - w: Width of the bounding region in pixels.
      - h: Height of the bounding region in pixels.
    """
    x: int = Field(..., ge=0, description="Top-left X pixel coordinate")
    y: int = Field(..., ge=0, description="Top-left Y pixel coordinate")
    w: int = Field(..., gt=0, description="Width in pixels")
    h: int = Field(..., gt=0, description="Height in pixels")


class RecognizedLine(BaseModel):
    """Single recognized text line with spatial metadata and reading order."""
    line_id: int = Field(..., description="1-indexed line order")
    text: str = Field(..., description="Recognized line text in logical Unicode RTL order")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Line recognition confidence if supported")
    bbox: BoundingBox | None = Field(None, description="Bounding box of the segmented line")
    reading_order: int = Field(..., description="Reading sequence (top-to-bottom)")


class RecognitionModelMeta(BaseModel):
    """Model identifier and version attached to response."""
    id: str
    version: str


class RecognitionResponse(BaseModel):
    """Structured response for recognition request."""
    status: str = Field(default="success", description="Request outcome status")
    request_id: str = Field(..., description="Unique trace identifier (UUID4)")
    text: str = Field(..., description="Full recognized manuscript text in logical Unicode order")
    language: LanguageEnum = Field(..., description="Processed manuscript language")
    model: RecognitionModelMeta = Field(..., description="Executing model metadata")
    processing_time_ms: float = Field(..., description="Actual elapsed inference duration in milliseconds")
    confidence: float | None = Field(None, description="Average confidence score (null if unsupported)")
    lines: list[RecognizedLine] = Field(default_factory=list, description="Line-level segmentation results")
    warnings: list[str] = Field(default_factory=list, description="Warnings or limitations encountered")
