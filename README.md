# Persian & Arabic Manuscript HTR/OCR AI Service

**Intelligent HTR/OCR System for Recognition, Reading, Evaluation, and Search in Persian and Arabic Manuscripts with Islamic Calligraphy.**

This production-grade AI microservice implements handwritten text recognition (HTR) tailored specifically to historical Islamic, Persian, and Arabic calligraphic manuscripts (e.g. Nasta'liq, Thuluth, Naskh, Shikasteh). It provides a decoupled backend architecture intended to integrate directly with a NestJS application.

---

## 1. System Architecture

```
                                  +-----------------------------+
                                  |   NestJS Platform Backend   |
                                  |   (Auth, Metadata, Storage) |
                                  +--------------+--------------+
                                                 |
                               HTTP / REST (JSON + Multipart)
                                                 |
+------------------------------------------------v-----------------------------------------------+
|                             Python HTR/OCR AI Service (FastAPI)                                |
|                                                                                                |
|   +-----------------------+   +------------------------+   +-------------------------------+   |
|   |   API Routes Layer    |   |  Preprocessing Service |   |     Model Registry Layer      |   |
|   |  - /health & /ready   |   |  - Grayscale & EXIF    |   |  - TorchCRNNEngine            |   |
|   |  - /v1/models         |   |  - Profile Deskew      |   |  - KrakenEngine (Adapter)     |   |
|   |  - /v1/recognize      |   |  - Ink Contrast Stretch|   |  - DemoEngine (Test/Mock)     |   |
|   |  - /v1/evaluate       |   |  - Median Despeckle    |   |  - Checkpoint Discovery       |   |
|   |  - /v1/train          |   |  - Otsu Binarization   |   +---------------+---------------+   |
|   +-----------+-----------+   |  - Line Valley Cutter  |                   |                   |
|               |               +-----------+------------+                   |                   |
|   +-----------v-----------+               |                                |                   |
|   |   Security & Storage  |               +----------------v---------------+                   |
|   |  - Path Traversal Blk |                                |                                   |
|   |  - Max 25MB / 100Mpx  |               +----------------v---------------+                   |
|   |  - JSONL Manifests    |               |    PyTorch CRNN Core Engine    |                   |
|   +-----------------------+               |  - VGG Feature Extractor       |                   |
|                                           |  - Bidirectional LSTM Sequence |                   |
|                                           |  - CTC Decoder + Confidence    |                   |
|                                           +--------------------------------+                   |
+------------------------------------------------------------------------------------------------+
```

---

## 2. Key Features

- **Decoupled Architecture**: Strictly isolates compute-heavy deep learning inference and training from the NestJS user/document backend.
- **Native PyTorch CRNN Engine**: Convolutional Recurrent Neural Network (CNN + Bidirectional LSTM + CTC Loss) designed for variable-width text lines.
- **Full Persian & Arabic Unicode Preservation**:
  - Persian specific letters: گ, چ, پ, ژ, ک, ی
  - Arabic letters: ة, ى, ئ, ؤ, إ, أ, آ, ء
  - Harakat / Diacritics: Fathah, Dammah, Kasrah, Sukun, Shaddah, Tanween
  - Eastern Arabic-Indic & Persian numerals (۰-۹ and ٠-٩)
  - Logical Unicode order preserved strictly without string reversal.
- **Configurable Preprocessing**:
  - Image decoding with EXIF transposition and dimension validation.
  - Projection profile variance deskewing.
  - Calligraphy ink contrast enhancement.
  - Median filter denoising to preserve delicate diacritics.
  - Adaptive Otsu binarization.
  - Text-line valley segmentation and bounding-box extraction.
- **Standardized Evaluation Metrics**:
  - Exact Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance.
  - Full breakdown of Substitutions ($S$), Deletions ($D$), and Insertions ($I$).
  - Per-language metrics compilation.
  - Optional search/evaluation normalization (Kaf/Yeh unification, Alif harmonization).
- **Asynchronous Training Dispatcher**:
  - Background training executor returning HTTP 202 with job tracking.
  - Reproducible random seeds.
  - Continuous validation CER tracking and best checkpoint preservation.
- **Security Hardened**:
  - Strict path traversal prevention: all file references restricted strictly inside approved root directories.
  - Upload file size limits (25MB) and pixel resolution caps (100 Megapixels).
  - Unprivileged Docker non-root user (`appuser`).

---

## 3. Directory Structure

```
.
├── app/
│   ├── main.py                     # FastAPI application entry point
│   ├── api/
│   │   └── routes/
│   │       ├── health.py           # /health and /ready probes
│   │       ├── models.py           # /v1/models registry endpoints
│   │       ├── recognition.py      # /v1/recognize endpoint
│   │       ├── evaluation.py       # /v1/evaluate benchmark endpoint
│   │       └── training.py         # /v1/train job endpoints
│   ├── core/
│   │   ├── config.py               # Pydantic Settings & environment
│   │   ├── logging.py              # Structured context logging
│   │   ├── exceptions.py           # Domain exceptions & HTTP mappings
│   │   └── security.py             # Safe path validation & sanitization
│   ├── schemas/
│   │   ├── health.py               # Health & Readiness schemas
│   │   ├── models.py               # Model registry schemas
│   │   ├── recognition.py          # Recognition request/response schemas
│   │   ├── evaluation.py           # CER/WER evaluation schemas
│   │   └── training.py             # Training job schemas
│   ├── services/
│   │   ├── recognition_service.py  # Inference orchestration
│   │   ├── preprocessing_service.py# Image enhancement & deskewing
│   │   ├── model_registry.py       # Engine lifecycle & model discovery
│   │   ├── dataset_service.py      # Manifest validation & integrity
│   │   ├── evaluation_service.py   # CER/WER computation
│   │   └── training_service.py     # Asynchronous training runner
│   ├── engines/
│   │   ├── base.py                 # Abstract BaseEngine interface
│   │   ├── torch_engine.py         # Native PyTorch CRNN Engine
│   │   ├── kraken_engine.py        # Kraken adapter & verification
│   │   └── demo_engine.py          # Deterministic test engine
│   └── ml/
│       ├── model_arch.py           # PyTorch CRNN architecture
│       ├── persian_arabic.py       # Unicode vocabulary, CTC decode, normalization
│       ├── metrics.py              # Levenshtein CER and WER algorithms
│       ├── dataset.py              # PyTorch Dataset & dynamic batch collation
│       ├── trainer.py              # PyTorch CTCLoss training loop
│       └── checkpoints.py          # State dict & metadata persistence
├── storage/
│   ├── models/                     # Production model checkpoints (.pt)
│   ├── checkpoints/                # Training run checkpoints
│   └── datasets/                   # JSONL manifests and manuscript images
├── tests/                          # 26 automated unit & integration tests
├── scripts/
│   ├── prepare_dataset.py          # Synthetic manuscript generator
│   ├── train.py                    # CLI training entry point
│   ├── evaluate.py                 # CLI evaluation entry point
│   └── run_dev.sh                  # Development startup script
├── Dockerfile                      # Production container image
├── docker-compose.yml              # Local compose orchestration
├── requirements.txt                # Pinned dependencies
├── pyproject.toml                  # Ruff and Pytest configuration
├── .env.example                    # Documented configuration template
└── README.md                       # Comprehensive documentation
```

---

## 4. Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Runtime environment (`development`, `production`, `testing`) |
| `HOST` | `0.0.0.0` | Bind host address |
| `PORT` | `8001` | Service port (8000 inside Docker, 8001 on host) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_ENGINE` | `torch_crnn` | Active default engine: `torch_crnn`, `kraken`, `demo` |
| `DEFAULT_MODEL_ID` | `persian-arabic-crnn-v1` | Identifier for active default model |
| `MODEL_DIR` | `./storage/models` | Filesystem root for loaded model weights |
| `CHECKPOINT_DIR` | `./storage/checkpoints` | Filesystem root for training checkpoints |
| `DATASET_ROOT` | `./storage/datasets` | Filesystem root for labeled dataset manifests |
| `MAX_UPLOAD_SIZE` | `26214400` | Max upload size in bytes (25 MB) |
| `MAX_IMAGE_PIXELS` | `100000000` | Maximum allowable image pixel count (100 Mpx) |
| `INFERENCE_DEVICE` | `cpu` | Device target: `cpu` (default) or `cuda` |
| `ALLOWED_ORIGINS` | `*` | Allowed CORS origins |

---

## 5. Quick Start (Local Development)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Dataset Fixture
```bash
python3 scripts/prepare_dataset.py
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

### 4. Run Linter
```bash
ruff check app/ tests/ scripts/
```

### 5. Launch the Service
```bash
./scripts/run_dev.sh
# Service runs at http://localhost:8001
# Interactive Swagger docs: http://localhost:8001/docs
```

---

## 6. Docker Deployment

### Build and Run with Docker Compose
```bash
docker-compose up -d --build
```

### Build Image Directly
```bash
docker build -t persian-arabic-htr-service:latest .
docker run -p 8000:8000 \
  -v $(pwd)/storage/models:/app/storage/models \
  -v $(pwd)/storage/datasets:/app/storage/datasets \
  persian-arabic-htr-service:latest
```

---

## 7. HTTP API Reference

### Health Check (Liveness Probe)
```http
GET /health
```
```json
{
  "status": "ok",
  "version": "1.0.0",
  "ready": true,
  "timestamp": "2026-10-30T08:35:01Z"
}
```

### Readiness Check (Model & Dependencies Probe)
```http
GET /ready
```
```json
{
  "ready": true,
  "service_status": "ready",
  "default_model_status": "MODEL_READY",
  "dependencies": {
    "pytorch": { "version": "2.15.0+cpu", "cuda_available": false, "device": "cpu" },
    "storage": { "model_dir_exists": true, "dataset_root_exists": true, "checkpoint_dir_exists": true },
    "default_model": { "id": "persian-arabic-crnn-v1", "status": "MODEL_READY", "ready": true }
  },
  "timestamp": "2026-10-30T08:35:01Z"
}
```

### List Models
```http
GET /v1/models
```

### Recognize Manuscript Image
```http
POST /v1/recognize
Content-Type: multipart/form-data
```
**Form Fields:**
- `file`: Raw image binary (JPEG, PNG, TIFF, WebP).
- `image_ref`: *(Optional alternative)* Relative path inside `DATASET_ROOT`.
- `language`: `PERSIAN`, `ARABIC`, or `MIXED`.
- `model_id`: *(Optional)* Model ID to execute.
- `deskew`: Boolean (default `false`).
- `contrast`: Boolean (default `false`).
- `denoise`: Boolean (default `false`).
- `binarize`: Boolean (default `false`).
- `layout_analysis`: Boolean (default `true`).

**Response (200 OK):**
```json
{
  "status": "success",
  "request_id": "fe3a2457-978c-4b40-ab6d-f03a72a6a313",
  "text": "به نام خداوند جان و خرد\nکزین برتر اندیشه برنگذرد",
  "language": "PERSIAN",
  "model": {
    "id": "persian-arabic-crnn-v1",
    "version": "1.0.0"
  },
  "processing_time_ms": 28.4,
  "confidence": 0.942,
  "lines": [
    {
      "line_id": 1,
      "text": "به نام خداوند جان و خرد",
      "confidence": 0.942,
      "bbox": { "x": 0, "y": 0, "w": 500, "h": 32 },
      "reading_order": 1
    },
    {
      "line_id": 2,
      "text": "کزین برتر اندیشه برنگذرد",
      "confidence": 0.915,
      "bbox": { "x": 0, "y": 32, "w": 500, "h": 32 },
      "reading_order": 2
    }
  ],
  "warnings": ["Applied preprocessing: grayscale, contrast_enhancement"]
}
```

### Evaluate Dataset Manifest (CER / WER)
```http
POST /v1/evaluate
Content-Type: application/json
```
```json
{
  "dataset_manifest": "sample_persian_arabic/test.jsonl",
  "model_id": "persian-arabic-crnn-v1",
  "split": "test",
  "normalize_text": false
}
```
**Response (200 OK):**
```json
{
  "dataset_manifest": "sample_persian_arabic/test.jsonl",
  "model_id": "persian-arabic-crnn-v1",
  "model_version": "1.0.0",
  "metrics": {
    "cer": 0.082,
    "wer": 0.125,
    "total_samples": 2,
    "evaluated_samples": 2,
    "skipped_samples": 0,
    "total_chars": 74,
    "char_substitutions": 4,
    "char_deletions": 1,
    "char_insertions": 1,
    "total_words": 16,
    "word_errors": 2
  },
  "per_language_results": {
    "PERSIAN": { "cer": 0.052, "wer": 0.083, "samples": 1, "total_chars": 38 },
    "MIXED": { "cer": 0.111, "wer": 0.166, "samples": 1, "total_chars": 36 }
  }
}
```

### Start Model Training Run
```http
POST /v1/train
Content-Type: application/json
```
```json
{
  "model_name": "persian-crnn-v2",
  "train_manifest": "sample_persian_arabic/train.jsonl",
  "val_manifest": "sample_persian_arabic/val.jsonl",
  "epochs": 5,
  "batch_size": 4,
  "learning_rate": 0.001
}
```
**Response (202 Accepted):**
```json
{
  "job_id": "job-b46c5129",
  "status": "RUNNING",
  "model_name": "persian-crnn-v2",
  "total_epochs": 5,
  "started_at": "2026-10-30T08:35:14Z"
}
```

---

## 8. CLI Commands

### Command-Line Model Training
```bash
python3 scripts/train.py \
  --model-name persian-arabic-crnn-v1 \
  --train-manifest sample_persian_arabic/train.jsonl \
  --val-manifest sample_persian_arabic/val.jsonl \
  --epochs 5 \
  --batch-size 4
```

### Command-Line Model Evaluation
```bash
python3 scripts/evaluate.py \
  --manifest sample_persian_arabic/test.jsonl \
  --model-id persian-arabic-crnn-v1 \
  --out storage/evaluation_report.json
```

---

## 9. Known Constraints & Calligraphy Notes

1. **Sloping Baselines**: Persian Nasta'liq and Shikasteh scripts feature diagonally sloping baselines where characters cascade from upper-right to lower-left. Projection-profile valley segmentation requires bounding boxes that enclose the full cascading cluster.
2. **Interlinear Overlaps**: Sub-baseline descenders (e.g., elongated 'ی' or 'ن') frequently intersect with ascenders ('ا', 'ل') of subsequent lines. Non-aggressive binarization and grayscale retention preserve these intersections.
3. **Diacritics & Dots**: Vowel markings (Harakat) and letter dots (Nuqteh) are essential for distinguishing Persian/Arabic homographs (ب / ت / ث / ن / ی). Aggressive binarization or erosion filters must be avoided.
4. **Honest Readiness Reporting**: If model checkpoints are not found in `storage/models`, the service cleanly returns `MODEL_NOT_CONFIGURED` (HTTP 503) without generating fabricated pseudo-transcriptions.
