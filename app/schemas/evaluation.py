"""Evaluation request and response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Configuration for running an evaluation against a labeled dataset manifest."""
    dataset_manifest: str = Field(..., description="Path to dataset manifest (.jsonl) relative to DATASET_ROOT")
    model_id: str | None = Field(None, description="Model to evaluate (defaults to DEFAULT_MODEL_ID)")
    split: str | None = Field("test", description="Dataset split to evaluate ('test', 'val', or 'all')")
    normalize_text: bool = Field(False, description="Whether to apply search/evaluation text normalization")
    max_samples: int | None = Field(None, ge=1, description="Optional cap on number of samples to evaluate")


class EvaluationMetricSummary(BaseModel):
    """Standard HTR recognition metrics."""
    cer: float = Field(..., description="Character Error Rate: (S + D + I) / N_chars")
    wer: float = Field(..., description="Word Error Rate: (S_w + D_w + I_w) / N_words")
    total_samples: int = Field(..., description="Total samples in manifest split")
    evaluated_samples: int = Field(..., description="Samples successfully evaluated")
    skipped_samples: int = Field(..., description="Samples skipped due to missing files or errors")
    total_chars: int = Field(..., description="Total ground truth character count")
    char_substitutions: int = Field(..., description="Character substitutions")
    char_deletions: int = Field(..., description="Character deletions")
    char_insertions: int = Field(..., description="Character insertions")
    total_words: int = Field(..., description="Total ground truth word count")
    word_errors: int = Field(..., description="Word-level edit distance errors")


class SamplePredictionResult(BaseModel):
    """Detailed prediction vs ground truth comparison for a sample."""
    sample_id: str
    ground_truth: str
    prediction: str
    cer: float
    wer: float


class EvaluationResponse(BaseModel):
    """Structured evaluation report."""
    dataset_manifest: str
    model_id: str
    model_version: str
    normalized: bool
    split: str
    metrics: EvaluationMetricSummary
    per_language_results: dict[str, Any] = Field(default_factory=dict)
    sample_results: list[SamplePredictionResult] = Field(default_factory=list)
    created_at: str
