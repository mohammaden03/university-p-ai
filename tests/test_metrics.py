"""
Tests for CER, WER, and Persian/Arabic script normalization.
"""

from app.ml.metrics import calculate_cer, calculate_wer
from app.ml.persian_arabic import normalize_persian_arabic


def test_cer_exact_match():
    """Verify CER is 0.0 for identical strings."""
    text = "به نام خداوند جان و خرد"
    cer, s, d, ins, n = calculate_cer(text, text)
    assert cer == 0.0
    assert s == 0 and d == 0 and ins == 0
    assert n == len(text)


def test_cer_single_substitution():
    """Verify single character substitution calculation."""
    ref = "کتاب"  # 4 chars: ک, ت, ا, ب
    hyp = "کباب"  # 'ت' substituted with 'ب'
    cer, s, d, ins, n = calculate_cer(ref, hyp)
    assert s == 1
    assert d == 0
    assert ins == 0
    assert n == 4
    assert cer == 0.25


def test_cer_deletion_and_insertion():
    """Verify deletions and insertions."""
    ref = "سلام"   # 4 chars
    hyp = "سلا"    # 1 deletion ('م')
    cer, s, d, ins, n = calculate_cer(ref, hyp)
    assert d == 1 and ins == 0
    assert cer == 0.25

    hyp_ins = "سلاما"  # 1 insertion ('ا')
    cer2, s2, d2, ins2, n2 = calculate_cer(ref, hyp_ins)
    assert ins2 == 1 and d2 == 0
    assert cer2 == 0.25


def test_wer_calculation():
    """Verify Word Error Rate on Persian sentences."""
    ref = "به نام خداوند جان و خرد"  # 6 words
    hyp = "به نام پروردگار جان و خرد"  # 1 word substituted ('خداوند' -> 'پروردگار')
    wer, errors, total_words = calculate_wer(ref, hyp)
    assert total_words == 6
    assert errors == 1
    assert round(wer, 4) == round(1.0 / 6.0, 4)


def test_persian_arabic_normalization():
    """Verify script unification and optional diacritics removal."""
    # Arabic Yeh vs Persian Yeh and Arabic Kaf vs Persian Kaf
    arabic_text = "كتابٌ جميلٌ في اللُّغَةِ"
    normalized = normalize_persian_arabic(arabic_text, remove_diacritics=True, unify_kaf_yeh=True)

    # Arabic 'ك' -> 'ک'
    assert "ك" not in normalized
    assert "ک" in normalized

    # Arabic 'ي' -> 'ی'
    assert "ي" not in normalized
    assert "ی" in normalized

    # Harakat removed
    assert "\u064b" not in normalized  # Tanween
    assert "\u0651" not in normalized  # Shaddah
    assert "\u0650" not in normalized  # Kasrah
