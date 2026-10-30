"""
Model Registry service.
Discovers, instantiates, and manages lifecycles of HTR engines and models.
"""

from pathlib import Path

from app.core.config import settings
from app.core.exceptions import ModelNotFoundError
from app.core.logging import logger
from app.engines.base import BaseEngine
from app.engines.demo_engine import DemoEngine
from app.engines.kraken_engine import KrakenEngine
from app.engines.torch_engine import TorchCRNNEngine
from app.schemas.models import ModelInfo


class ModelRegistry:
    """Central registry tracking all configured recognition engines and models."""

    def __init__(self):
        self._engines: dict[str, BaseEngine] = {}
        self.default_model_id = settings.DEFAULT_MODEL_ID
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize engine adapters based on application configuration."""
        # 1. Native PyTorch CRNN Engine
        torch_engine = TorchCRNNEngine(
            model_id="persian-arabic-crnn-v1",
            device=settings.INFERENCE_DEVICE,
        )
        # Search for available checkpoints in MODEL_DIR and CHECKPOINT_DIR
        candidate_ckpts = [
            settings.MODEL_DIR / "persian_arabic_crnn.pt",
            settings.CHECKPOINT_DIR / "persian-arabic-crnn-v2_best.pt",
            settings.CHECKPOINT_DIR / "persian_arabic_best.pt",
        ]
        for ckpt in candidate_ckpts:
            if ckpt.exists():
                logger.info(f"Discovered candidate PyTorch model checkpoint at {ckpt}")
                try:
                    torch_engine.load_model("persian-arabic-crnn-v1", ckpt)
                    break
                except Exception as e:
                    logger.warning(f"Could not load discovered checkpoint {ckpt}: {e}")

        self._engines["persian-arabic-crnn-v1"] = torch_engine

        # 2. Kraken Historical HTR Engine
        kraken_engine = KrakenEngine(
            model_id="kraken-arabic-persian-v1",
            device=settings.INFERENCE_DEVICE,
        )
        kraken_candidate = settings.MODEL_DIR / "arabic_persian.mlmodel"
        if kraken_candidate.exists():
            kraken_engine.load_model("kraken-arabic-persian-v1", kraken_candidate)
        self._engines["kraken-arabic-persian-v1"] = kraken_engine

        # 3. Deterministic Demo / Test Engine
        demo_engine = DemoEngine(
            model_id="demo-persian-naskh-v1",
            device="cpu",
        )
        self._engines["demo-persian-naskh-v1"] = demo_engine

        # If configured engine is demo, set default model accordingly
        if settings.MODEL_ENGINE == "demo":
            self.default_model_id = "demo-persian-naskh-v1"
        elif settings.MODEL_ENGINE == "kraken":
            self.default_model_id = "kraken-arabic-persian-v1"

        logger.info(
            f"ModelRegistry initialized with {len(self._engines)} engines. "
            f"Default model: '{self.default_model_id}'."
        )

    def get_engine(self, model_id: str | None = None) -> BaseEngine:
        """Retrieve configured engine by model ID or return default."""
        target_id = model_id or self.default_model_id
        if target_id not in self._engines:
            raise ModelNotFoundError(target_id)
        return self._engines[target_id]

    def list_models(self) -> list[ModelInfo]:
        """Return information for all registered models."""
        return [engine.get_model_info() for engine in self._engines.values()]

    def reload_checkpoint(self, model_id: str, checkpoint_path: Path) -> ModelInfo:
        """Reload or update an engine with a newly trained checkpoint."""
        engine = self.get_engine(model_id)
        engine.load_model(model_id, checkpoint_path)
        return engine.get_model_info()


registry = ModelRegistry()
