from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_RE = re.compile(
    r"(?:####|final answer(?:\s+is)?|the answer is)\s*[:=]?\s*([-$+]?[0-9][0-9,./]*)",
    re.I,
)
_NUMBER_RE = re.compile(r"[-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?(?:/[0-9]+)?")


def _canonical_number(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip().replace("$", "").replace(",", "").rstrip(".")
    try:
        if "/" in cleaned:
            value = Fraction(cleaned)
            return f"{value.numerator}/{value.denominator}"
        value = Decimal(cleaned)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    normalized = value.normalize()
    return format(normalized, "f")


def extract_final_number(text: str) -> str | None:
    matches = _FINAL_RE.findall(text)
    if matches:
        return _canonical_number(matches[-1])
    boxed = _BOXED_RE.findall(text)
    if boxed:
        candidate_numbers = _NUMBER_RE.findall(boxed[-1])
        if candidate_numbers:
            return _canonical_number(candidate_numbers[-1])
    numbers = _NUMBER_RE.findall(text)
    return _canonical_number(numbers[-1]) if numbers else None


def gsm8k_exact_match(prediction: str, reference: str) -> tuple[bool, str | None, str | None]:
    predicted = extract_final_number(prediction)
    gold = extract_final_number(reference)
    return predicted is not None and predicted == gold, predicted, gold
