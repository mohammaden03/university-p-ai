"""
Persian and Arabic script specifications, character vocabularies, and text normalization.

Design rules:
  1. All recognition outputs are strictly preserved in logical Unicode order.
  2. Strings are NEVER reversed manually (visual RTL rendering is a presentation concern).
  3. Diacritics and ligatures are preserved in primary transcription.
  4. Normalization is strictly optional for search indexing and evaluation matching.
"""

import re
import unicodedata

# CTC Blank token is at index 0
CTC_BLANK_TOKEN = "<BLANK>"

# Comprehensive Persian & Arabic Alphabet
# Includes standard Arabic alphabet, Persian extensions (گ چ پ ژ ک ی), ligatures, numerals, diacritics
PERSIAN_SPECIFIC_CHARS = ["پ", "چ", "ژ", "گ", "ک", "ی"]
ARABIC_BASE_CHARS = [
    "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش",
    "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ل", "م", "ن", "ه", "و", "ي", "ك",
    "ة", "ى", "آ", "أ", "إ", "ئ", "ؤ", "ء"
]

# Diacritics (Harakat / Tashkeel)
HARAKAT = [
    "\u064e",  # Fathah (َ)
    "\u064f",  # Dammah (ُ)
    "\u0650",  # Kasrah (ِ)
    "\u0652",  # Sukun (ْ)
    "\u0651",  # Shaddah (ّ)
    "\u064b",  # Fathatan / Tanween Fath (ً)
    "\u064c",  # Dammatan / Tanween Damm (ٌ)
    "\u064d",  # Kasratan / Tanween Kasr (ٍ)
    "\u0653",  # Maddah (ٓ)
    "\u0654",  # Hamza Above (ٔ)
    "\u0655",  # Hamza Below (ٕ)
]

# Numerals (Eastern Arabic & Persian)
NUMERALS = [
    "۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹",  # Persian
    "٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩",  # Arabic-Indic
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"   # Standard
]

# Common Punctuation & Whitespace
PUNCTUATION_AND_SPECIAL = [
    " ",       # Space
    "\u200c",  # Zero-Width Non-Joiner (ZWNJ / نیم‌فاصله)
    "ـ",       # Tatweel / Kashida (\u0640)
    ".", "،", "؛", "؟", "!", ":", "«", "»", "(", ")", "[", "]", "-", "/"
]

# Combined canonical vocabulary
DEFAULT_VOCABULARY: list[str] = (
    [CTC_BLANK_TOKEN]
    + ARABIC_BASE_CHARS
    + PERSIAN_SPECIFIC_CHARS
    + HARAKAT
    + NUMERALS
    + PUNCTUATION_AND_SPECIAL
)

# Deduplicate while preserving order
_seen = set()
DEFAULT_VOCABULARY = [c for c in DEFAULT_VOCABULARY if not (c in _seen or _seen.add(c))]

CHAR_TO_IDX: dict[str, int] = {char: idx for idx, char in enumerate(DEFAULT_VOCABULARY)}
IDX_TO_CHAR: dict[int, str] = {idx: char for idx, char in enumerate(DEFAULT_VOCABULARY)}


def text_to_indices(text: str, char_to_idx: dict[str, int] = CHAR_TO_IDX) -> list[int]:
    """Convert Unicode text to token indices, ignoring unknown tokens."""
    indices = []
    for char in text:
        if char in char_to_idx and char != CTC_BLANK_TOKEN:
            indices.append(char_to_idx[char])
    return indices


def indices_to_text(indices: list[int], idx_to_char: dict[int, str] = IDX_TO_CHAR) -> str:
    """Convert indices back to raw text (before CTC collapse)."""
    return "".join(idx_to_char.get(idx, "") for idx in indices if idx != 0)


def ctc_greedy_decode(
    logits: list[list[float]],
    idx_to_char: dict[int, str] = IDX_TO_CHAR
) -> tuple[str, float]:
    """
    Greedy argmax CTC decoding for sequential time-step logits.
    Collapses adjacent identical tokens and removes blank tokens (index 0).
    Returns (decoded_text, average_confidence).
    """
    import math

    decoded_chars: list[str] = []
    confidences: list[float] = []
    prev_idx = -1

    for step_logits in logits:
        # Find argmax and softmax probability
        max_idx = 0
        max_val = step_logits[0]
        for idx, val in enumerate(step_logits):
            if val > max_val:
                max_val = val
                max_idx = idx

        # Compute softmax confidence for the chosen token
        exp_sum = sum(math.exp(v - max_val) for v in step_logits)
        conf = 1.0 / exp_sum if exp_sum > 0 else 0.0

        if max_idx != 0:  # Not blank
            if max_idx != prev_idx:  # Collapse duplicates
                char = idx_to_char.get(max_idx, "")
                if char and char != CTC_BLANK_TOKEN:
                    decoded_chars.append(char)
                    confidences.append(conf)

        prev_idx = max_idx

    decoded_text = "".join(decoded_chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return decoded_text, avg_conf


def normalize_persian_arabic(
    text: str,
    remove_diacritics: bool = False,
    unify_alif: bool = True,
    unify_kaf_yeh: bool = True,
    remove_tatweel: bool = True,
    normalize_numerals: bool = False,
) -> str:
    """
    Configurable normalization for Persian/Arabic text.
    Used for search index matching and standardized CER/WER evaluation.
    Primary recognition preserves the raw script as transcribed.
    """
    if not text:
        return ""

    # Unicode canonical decomposition and composition (NFC)
    text = unicodedata.normalize("NFC", text)

    # Remove Tatweel / Kashida (elongation line used in Islamic calligraphy)
    if remove_tatweel:
        text = text.replace("ـ", "")

    # Remove diacritics / harakat if requested
    if remove_diacritics:
        for diacritic in HARAKAT:
            text = text.replace(diacritic, "")

    # Unify Alif variants (آ / أ / إ -> ا)
    if unify_alif:
        text = re.sub("[آأإٱ]", "ا", text)

    # Unify Arabic/Persian letter variants
    if unify_kaf_yeh:
        # Arabic Kaf (ك) -> Persian Kaf (ک)
        text = text.replace("ك", "ک")
        # Arabic Yeh (ي / ى) -> Persian Yeh (ی)
        text = text.replace("ي", "ی").replace("ى", "ی")
        # Arabic Teh Marbuta (ة) -> Persian Heh (ه)
        text = text.replace("ة", "ه")

    # Normalize Eastern Arabic digits to Persian
    if normalize_numerals:
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        for a_d, p_d in zip(arabic_digits, persian_digits):
            text = text.replace(a_d, p_d)

    # Normalize excessive spaces and clean whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text
