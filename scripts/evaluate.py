"""
Command-line evaluation script for HTR models on labeled manuscript datasets.
Usage:
  python3 scripts/evaluate.py --manifest sample_persian_arabic/test.jsonl --model-id persian-arabic-crnn-v1
"""

import argparse
import sys
from pathlib import Path

# Ensure app package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.evaluation import EvaluationRequest
from app.services.evaluation_service import EvaluationService


def main():
    parser = argparse.ArgumentParser(description="Evaluate Persian/Arabic Manuscript HTR Model")
    parser.add_argument("--manifest", type=str, default="sample_persian_arabic/test.jsonl", help="Manifest relative to DATASET_ROOT")
    parser.add_argument("--model-id", type=str, default=None, help="Model ID (defaults to DEFAULT_MODEL_ID)")
    parser.add_argument("--split", type=str, default="test", help="Split name to evaluate ('test', 'val', 'all')")
    parser.add_argument("--normalize", action="store_true", help="Apply search/evaluation normalization")
    parser.add_argument("--out", type=str, default=None, help="Output JSON filepath for metrics report")
    args = parser.parse_args()

    print("==================================================================")
    print(" Persian & Arabic Manuscript HTR - Evaluation Report")
    print("==================================================================")
    print(f"Manifest:  {args.manifest}")
    print(f"Model ID:  {args.model_id or 'default'}")
    print(f"Split:     {args.split}")
    print(f"Normalize: {args.normalize}")

    req = EvaluationRequest(
        dataset_manifest=args.manifest,
        model_id=args.model_id,
        split=args.split,
        normalize_text=args.normalize,
    )

    resp = EvaluationService.evaluate_manifest(req)

    print("\n------------------------- Results --------------------------------")
    print(f"Model ID:            {resp.model_id} (version: {resp.model_version})")
    print(f"Evaluated Samples:   {resp.metrics.evaluated_samples} / {resp.metrics.total_samples}")
    print(f"Skipped Samples:     {resp.metrics.skipped_samples}")
    print(f"Total Chars:         {resp.metrics.total_chars}")
    print(f"Character Error Rate (CER): {resp.metrics.cer * 100:.2f}%")
    print(f"  - Substitutions:   {resp.metrics.char_substitutions}")
    print(f"  - Deletions:       {resp.metrics.char_deletions}")
    print(f"  - Insertions:      {resp.metrics.char_insertions}")
    print(f"Total Words:         {resp.metrics.total_words}")
    print(f"Word Error Rate (WER):      {resp.metrics.wer * 100:.2f}% (errors: {resp.metrics.word_errors})")

    if resp.per_language_results:
        print("\nPer-Language Breakdown:")
        for lang, l_m in resp.per_language_results.items():
            print(f"  [{lang}] CER: {l_m['cer'] * 100:.2f}% | WER: {l_m['wer'] * 100:.2f}% ({l_m['samples']} samples)")

    print("\nSample Comparisons (Ground Truth vs Prediction):")
    for s in resp.sample_results[:5]:
        print(f"  Sample ID: {s.sample_id}")
        print(f"    GT:   {s.ground_truth}")
        print(f"    PRED: {s.prediction}")
        print(f"    CER:  {s.cer:.4f} | WER: {s.wer:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(resp.model_dump_json(indent=2))
        print(f"\nSaved structured evaluation report to {out_path}")


if __name__ == "__main__":
    main()
