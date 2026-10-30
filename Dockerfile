# Multi-stage Dockerfile for Persian and Arabic Manuscript HTR/OCR AI Service
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    PORT=8000

# Install minimal OS dependencies for image processing and health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Install PyTorch CPU first to leverage Docker layer caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install pinned requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create storage mount points with permissions
RUN mkdir -p /app/storage/models /app/storage/checkpoints /app/storage/datasets /tmp/htr_service && \
    chown -R appuser:appuser /app /tmp/htr_service

# Copy application source and scripts
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser storage/ ./storage/
COPY --chown=appuser:appuser .env.example .env

USER appuser

EXPOSE 8000

# Container liveness check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
