"""
Evaluation metrics for Handwritten Text Recognition (HTR).
Implements Character Error Rate (CER) and Word Error Rate (WER) using
exact Levenshtein edit distance with full error category breakdowns.
"""

import re
from typing import Any


def levenshtein_distance_with_ops(
    ref: list[Any],
    hyp: list[Any]
) -> tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance and operation breakdown.
    Returns: (distance, substitutions, deletions, insertions)
    """
    n = len(ref)
    m = len(hyp)

    # dp[i][j] stores (cost, subst, del, ins)
    # Using 2D table for precise backtrace and op counting
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = (0, i, 0)  # deletions

    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = (0, 0, j)  # insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1

                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost

                if min_cost == sub_cost:
                    s, d, ins = ops[i - 1][j - 1]
                    ops[i][j] = (s + 1, d, ins)
                elif min_cost == del_cost:
                    s, d, ins = ops[i - 1][j]
                    ops[i][j] = (s, d + 1, ins)
                else:
                    s, d, ins = ops[i][j - 1]
                    ops[i][j] = (s, d, ins + 1)

    s, d, ins = ops[n][m]
    return dp[n][m], s, d, ins


def calculate_cer(
    reference: str,
    hypothesis: str
) -> tuple[float, int, int, int, int]:
    """
    Calculate Character Error Rate (CER).
    CER = (Substitutions + Deletions + Insertions) / Total Reference Characters
    Returns: (cer, substitutions, deletions, insertions, total_ref_chars)
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    n = len(ref_chars)

    if n == 0:
        if len(hyp_chars) == 0:
            return 0.0, 0, 0, 0, 0
        else:
            return 1.0, 0, 0, len(hyp_chars), 0

    dist, s, d, ins = levenshtein_distance_with_ops(ref_chars, hyp_chars)
    cer = dist / float(n)
    return cer, s, d, ins, n


def tokenize_persian_arabic_words(text: str) -> list[str]:
    """
    Tokenizes Persian and Arabic text into words.
    Preserves Persian/Arabic letters, diacritics, and ZWNJ.
    """
    # Split on whitespace while removing standalone punctuation tokens
    raw_tokens = re.findall(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\w\u200c]+", text)
    return [t.strip() for t in raw_tokens if t.strip()]


def calculate_wer(
    reference: str,
    hypothesis: str
) -> tuple[float, int, int]:
    """
    Calculate Word Error Rate (WER).
    WER = EditDistance(Words(reference), Words(hypothesis)) / len(Words(reference))
    Returns: (wer, word_errors, total_ref_words)
    """
    ref_words = tokenize_persian_arabic_words(reference)
    hyp_words = tokenize_persian_arabic_words(hypothesis)
    n = len(ref_words)

    if n == 0:
        if len(hyp_words) == 0:
            return 0.0, 0, 0
        else:
            return 1.0, len(hyp_words), 0

    dist, _, _, _ = levenshtein_distance_with_ops(ref_words, hyp_words)
    wer = dist / float(n)
    return wer, dist, n
