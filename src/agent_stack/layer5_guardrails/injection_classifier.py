"""
Layer 5: Guardrails -- the input half.

Prompt injection (text that makes the model ignore its instructions) is
listed as the top risk, LLM01, in OWASP's Top 10 for LLM Applications.
It comes in two forms:

  - Direct: typed straight into the chat by the user.
  - Indirect: hidden in content the agent reads on the user's behalf --
    a document, an email, a support ticket, a web page -- aimed at the
    agent rather than at whoever is reading the content.

`classify_injection` is applied to BOTH: to user input before it reaches
the model, and to retrieved/fetched text (RAG context, tool results)
before it is trusted, since an indirect attack lives in exactly that
text.

This is a small, explainable keyword/pattern classifier, not a trained
model -- it is meant to demonstrate the layer and its failure mode, not
to be production-grade. No filter here is claimed to be complete: the
goal of the overall architecture is that a successful injection still
can't do much, because the permission and confirmation layers (3 and 4)
are still standing even when this classifier misses something.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(
        r"(?i)\bignore (all |any )?(previous|prior|above|your) instructions?\b"
    )),
    ("disregard_rules", re.compile(
        r"(?i)\bdisregard (the |your )?(system prompt|rules|instructions)\b"
    )),
    ("role_override", re.compile(
        r"(?i)\byou are now\b.{0,40}\b(developer mode|unrestricted|jailbroken)\b"
    )),
    ("reveal_prompt", re.compile(
        r"(?i)\b(reveal|print|show|repeat) (your |the )?(system prompt|"
        r"instructions|hidden prompt)\b"
    )),
    ("fake_system_turn", re.compile(r"(?i)^\s*system\s*:")),
    ("embedded_directive_to_agent", re.compile(
        r"(?i)\b(assistant|agent|ai)[,:]?\s*(please\s+)?(ignore|forget|"
        r"override)\b"
    )),
]


@dataclass
class InjectionVerdict:
    is_suspicious: bool
    matched_rules: list[str]
    confidence: float  # 0.0 - 1.0, naive: fraction of rules matched

    def __bool__(self) -> bool:  # so `if classify_injection(text):` reads naturally
        return self.is_suspicious


def classify_injection(text: str) -> InjectionVerdict:
    matched = [name for name, pattern in _SUSPICIOUS_PATTERNS if pattern.search(text)]
    confidence = len(matched) / len(_SUSPICIOUS_PATTERNS)
    return InjectionVerdict(
        is_suspicious=bool(matched), matched_rules=matched, confidence=confidence
    )
