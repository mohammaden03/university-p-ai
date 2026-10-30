"""
Base recognition engine abstract interface.
All HTR engines (PyTorch, Kraken, Test/Demo) adhere strictly to this contract.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from app.schemas.models import ModelInfo, ModelStatus
from app.schemas.recognition import RecognizedLine


class BaseEngine(ABC):
    """Abstract interface for all HTR recognition engines."""

    @abstractmethod
    def load_model(self, model_id: str, checkpoint_path: Path | None = None) -> None:
        """
        Load and initialize model weights from checkpoint.
        Updates internal status to MODEL_READY or MODEL_LOAD_FAILED.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Return True ONLY if a valid model has been loaded and verified for inference.
        Never returns True on unconfigured or placeholder states.
        """

    @abstractmethod
    def get_status(self) -> ModelStatus:
        """Return the current logical model state."""

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return structured model metadata and capabilities."""

    @abstractmethod
    def recognize(
        self,
        image: Image.Image,
        language: str = "PERSIAN",
        options: dict[str, Any] | None = None,
    ) -> tuple[str, float | None, list[RecognizedLine], list[str]]:
        """
        Recognize text from an image.
        Returns:
            Tuple of:
              - Full recognized text in logical Unicode order
              - Average confidence score (or None if engine cannot compute true confidence)
              - List of recognized line objects with bounding boxes
              - List of warning strings
        """

    @abstractmethod
    def release(self) -> None:
        """Release GPU/CPU memory and unload model weights."""
