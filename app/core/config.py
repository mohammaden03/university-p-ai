"""Application Configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the Persian/Arabic Manuscript HTR service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Service
    APP_NAME: str = "Persian & Arabic Manuscript HTR/OCR AI Service"
    APP_VERSION: str = "1.0.0"
    APP_ENV: Literal["development", "production", "testing"] = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # Engine & Models
    MODEL_ENGINE: Literal["torch_crnn", "kraken", "demo"] = "torch_crnn"
    DEFAULT_MODEL_ID: str = "persian-arabic-crnn-v1"

    # File System Roots
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_DIR: Path = Field(default_factory=lambda: Path("./storage/models"))
    CHECKPOINT_DIR: Path = Field(default_factory=lambda: Path("./storage/checkpoints"))
    DATASET_ROOT: Path = Field(default_factory=lambda: Path("./storage/datasets"))
    TEMP_DIR: Path = Field(default_factory=lambda: Path("/tmp/htr_service"))

    # Security & Processing Limits
    MAX_UPLOAD_SIZE: int = 26_214_400  # 25 MB
    MAX_IMAGE_PIXELS: int = 100_000_000  # 100 Megapixels
    REQUEST_TIMEOUT_SECONDS: int = 60
    MAX_CONCURRENT_INFERENCES: int = 4
    INTERNAL_SERVICE_SECRET: str = "dev-secret-persian-htr-key"
    ALLOWED_ORIGINS: str = "*"

    # Hardware & Inference
    INFERENCE_DEVICE: Literal["cpu", "cuda"] = "cpu"

    def resolve_paths(self) -> None:
        """Resolve and ensure base directories exist safely."""
        self.MODEL_DIR = (self.BASE_DIR / self.MODEL_DIR).resolve() if not self.MODEL_DIR.is_absolute() else self.MODEL_DIR.resolve()
        self.CHECKPOINT_DIR = (self.BASE_DIR / self.CHECKPOINT_DIR).resolve() if not self.CHECKPOINT_DIR.is_absolute() else self.CHECKPOINT_DIR.resolve()
        self.DATASET_ROOT = (self.BASE_DIR / self.DATASET_ROOT).resolve() if not self.DATASET_ROOT.is_absolute() else self.DATASET_ROOT.resolve()
        self.TEMP_DIR = self.TEMP_DIR.resolve()

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.resolve_paths()
