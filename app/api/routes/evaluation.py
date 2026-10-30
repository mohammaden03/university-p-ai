"""
Evaluation endpoints.
Computes real Character Error Rate (CER) and Word Error Rate (WER) against labeled dataset manifests.
"""

from fastapi import APIRouter

from app.schemas.evaluation import EvaluationRequest, EvaluationResponse
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/v1/evaluate", tags=["Evaluation"])


@router.post("", response_model=EvaluationResponse)
async def evaluate_dataset(request: EvaluationRequest):
    """
    Evaluate an HTR model against a labeled ground-truth dataset manifest (.jsonl).
    Calculates exact CER and WER with Levenshtein edit distance and operation breakdown.
    """
    return EvaluationService.evaluate_manifest(request)
