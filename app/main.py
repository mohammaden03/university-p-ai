"""
Main FastAPI Application Entry Point.
Persian and Arabic Manuscript HTR/OCR AI Service.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import evaluation, health, models, recognition, training
from app.core.config import settings
from app.core.exceptions import HTRException
from app.core.logging import logger
from app.services.dataset_service import DatasetService
from app.services.model_registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info("Initializing Persian & Arabic Manuscript HTR Service...")
    logger.info(f"Environment: {settings.APP_ENV} | Host: {settings.HOST}:{settings.PORT}")
    logger.info(f"Storage Roots -> Models: {settings.MODEL_DIR} | Datasets: {settings.DATASET_ROOT}")
    logger.info(f"Default Engine: {settings.MODEL_ENGINE} | Default Model ID: {registry.default_model_id}")

    yield

    logger.info("Shutting down HTR Service and releasing model resources...")
    for model_id, engine in registry._engines.items():
        try:
            engine.release()
        except Exception:
            pass


app = FastAPI(
    title="Persian & Arabic Manuscript HTR/OCR AI Service",
    description=(
        "Intelligent HTR/OCR System for Recognition, Reading, and Advanced Search "
        "in Persian and Arabic Manuscripts with Islamic Calligraphy. "
        "Exposes HTTP API for image recognition, model registry, evaluation (CER/WER), "
        "dataset management, and reproducible PyTorch CRNN model training."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Inject unique request ID and measure processing duration."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id

    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    return response


@app.exception_handler(HTRException)
async def htr_exception_handler(request: Request, exc: HTRException):
    """Structured handler for domain-level HTR exceptions."""
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "request_id": req_id,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic request parameter validation failures."""
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "request_id": req_id,
            "error_code": "VALIDATION_ERROR",
            "message": "Request parameter validation failed.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Fallback handler for uncaught server errors."""
    req_id = getattr(request.state, "request_id", "UNKNOWN")
    logger.error(f"Unhandled exception [req:{req_id}]: {exc!s}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "request_id": req_id,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred while processing the manuscript request.",
            "details": {"error_type": type(exc).__name__},
        },
    )


# Dataset verification helper endpoint
@app.get("/v1/datasets/validate", tags=["Datasets"])
async def validate_dataset_endpoint(manifest_path: str):
    """Audit and validate a dataset manifest (.jsonl) within DATASET_ROOT."""
    return DatasetService.validate_dataset(manifest_path, verify_images=True)


# Mount routers
app.include_router(health.router)
app.include_router(models.router)
app.include_router(recognition.router)
app.include_router(evaluation.router)
app.include_router(training.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
