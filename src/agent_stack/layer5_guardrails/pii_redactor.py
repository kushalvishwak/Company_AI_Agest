"""
Layer 5: Guardrails -- the output half.

Output is scanned for PII before it reaches the screen, so salaries,
phone numbers, and ID numbers get redacted even if the input classifier
upstream missed the attempt that tried to surface them. The same
function is reused by Layer 7's tracer to redact logs -- a log is just
another place sensitive data can leak.

This is a small set of illustrative regexes, not a comprehensive PII
detector; it covers the categories the architecture write-up this
project implements names explicitly (salaries, phone numbers, IDs).
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\d{10}(?!\d)")),
    (
        "SALARY",
        re.compile(
            r"(?i)\b(?:salary(?:[_\s-]?inr)?|compensation)\b\s*"
            r"(?:(?:is|was|of)\s*)?(?:₹|rs\.?|inr)?\s?"
            r"[\d,]+(?:\.\d+)?\s?(?:lpa|lakhs?|per annum)?"
        ),
    ),
    ("ID_NUMBER", re.compile(r"\b(?:emp-|ssn-|aadhaar-)?\d{4,}\b", re.IGNORECASE)),
]


def redact_pii(text: str) -> str:
    redacted = text
    for label, pattern in _PATTERNS:
        redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
    return redacted
