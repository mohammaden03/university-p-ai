"""
Evaluation service executing CER and WER benchmarks on labeled manuscript datasets.
"""

from datetime import UTC, datetime
from typing import Any

from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.ml.metrics import calculate_cer, calculate_wer
from app.ml.persian_arabic import normalize_persian_arabic
from app.schemas.evaluation import (
    EvaluationMetricSummary,
    EvaluationRequest,
    EvaluationResponse,
    SamplePredictionResult,
)
from app.services.dataset_service import DatasetService
from app.services.model_registry import registry


class EvaluationService:
    """Runs reproducible evaluation metrics on labeled HTR test datasets."""

    @classmethod
    def evaluate_manifest(cls, request: EvaluationRequest) -> EvaluationResponse:
        # Load samples
        all_samples = DatasetService.load_manifest(request.dataset_manifest)

        # Filter by split if requested
        if request.split and request.split.lower() != "all":
            target_split = request.split.lower()
            samples = [s for s in all_samples if s.get("split", "test").lower() == target_split]
        else:
            samples = all_samples

        if request.max_samples and request.max_samples < len(samples):
            samples = samples[: request.max_samples]

        # Get model engine
        target_model_id = request.model_id or registry.default_model_id
        engine = registry.get_engine(target_model_id)
        model_info = engine.get_model_info()

        if not engine.is_ready():
            from app.core.exceptions import ModelNotConfiguredError
            raise ModelNotConfiguredError(
                target_model_id,
                f"Cannot evaluate model '{target_model_id}': engine is not in MODEL_READY state "
                f"(current: {model_info.status.value})."
            )

        total_samples = len(samples)
        evaluated_samples = 0
        skipped_samples = 0

        tot_chars = 0
        tot_subst = 0
        tot_del = 0
        tot_ins = 0

        tot_words = 0
        tot_word_errors = 0

        per_lang: dict[str, dict[str, Any]] = {}
        sample_results: list[SamplePredictionResult] = []

        for sample in samples:
            sample_id = str(sample.get("id", f"sample_{evaluated_samples + 1}"))
            img_rel = sample.get("image_path")
            gt_text = sample.get("text", "")
            lang = sample.get("language", "PERSIAN").upper()

            full_img_path = (settings.DATASET_ROOT / img_rel).resolve()
            if not full_img_path.exists():
                logger.warning(f"Evaluation skipped sample '{sample_id}': image missing at {img_rel}")
                skipped_samples += 1
                continue

            try:
                with Image.open(full_img_path) as im:
                    # Optional crop if bbox is present in line-level sample
                    if sample.get("bbox"):
                        b = sample["bbox"]
                        im = im.crop((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))

                    pred_text, _, _, _ = engine.recognize(im, language=lang)
            except Exception as e:
                logger.warning(f"Evaluation inference failed on sample '{sample_id}': {e}")
                skipped_samples += 1
                continue

            # Optional normalization for standardized matching
            ref = normalize_persian_arabic(gt_text) if request.normalize_text else gt_text
            hyp = normalize_persian_arabic(pred_text) if request.normalize_text else pred_text

            # Compute CER
            s_cer, s, d, ins, n_chars = calculate_cer(ref, hyp)
            # Compute WER
            s_wer, w_errs, n_words = calculate_wer(ref, hyp)

            tot_chars += n_chars
            tot_subst += s
            tot_del += d
            tot_ins += ins

            tot_words += n_words
            tot_word_errors += w_errs

            evaluated_samples += 1

            # Per-language stats
            if lang not in per_lang:
                per_lang[lang] = {"chars": 0, "errors": 0, "words": 0, "word_errors": 0, "samples": 0}
            per_lang[lang]["chars"] += n_chars
            per_lang[lang]["errors"] += (s + d + ins)
            per_lang[lang]["words"] += n_words
            per_lang[lang]["word_errors"] += w_errs
            per_lang[lang]["samples"] += 1

            if len(sample_results) < 50:  # Keep first 50 for inspected samples
                sample_results.append(
                    SamplePredictionResult(
                        sample_id=sample_id,
                        ground_truth=ref,
                        prediction=hyp,
                        cer=round(s_cer, 4),
                        wer=round(s_wer, 4),
                    )
                )

        global_cer = (tot_subst + tot_del + tot_ins) / float(tot_chars) if tot_chars > 0 else 0.0
        global_wer = tot_word_errors / float(tot_words) if tot_words > 0 else 0.0

        # Compile per-language CER/WER
        lang_summary: dict[str, Any] = {}
        for l_name, l_data in per_lang.items():
            l_cer = l_data["errors"] / float(l_data["chars"]) if l_data["chars"] > 0 else 0.0
            l_wer = l_data["word_errors"] / float(l_data["words"]) if l_data["words"] > 0 else 0.0
            lang_summary[l_name] = {
                "cer": round(l_cer, 4),
                "wer": round(l_wer, 4),
                "samples": l_data["samples"],
                "total_chars": l_data["chars"],
            }

        metric_summary = EvaluationMetricSummary(
            cer=round(global_cer, 4),
            wer=round(global_wer, 4),
            total_samples=total_samples,
            evaluated_samples=evaluated_samples,
            skipped_samples=skipped_samples,
            total_chars=tot_chars,
            char_substitutions=tot_subst,
            char_deletions=tot_del,
            char_insertions=tot_ins,
            total_words=tot_words,
            word_errors=tot_word_errors,
        )

        return EvaluationResponse(
            dataset_manifest=request.dataset_manifest,
            model_id=model_info.id,
            model_version=model_info.version,
            normalized=request.normalize_text,
            split=request.split or "test",
            metrics=metric_summary,
            per_language_results=lang_summary,
            sample_results=sample_results,
            created_at=datetime.now(UTC).isoformat(),
        )
