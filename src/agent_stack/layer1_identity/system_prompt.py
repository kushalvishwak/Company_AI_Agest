"""
Layer 1: API + System Prompt
=============================

Day one of most internal AI projects is a model API with nothing telling
it who it is supposed to be. Left alone, the model answers as whatever
persona its training leaned toward, and introduces itself as its vendor's
product rather than "the company's assistant".

The system prompt fixes that: a block of hidden instructions sent before
every conversation, setting the assistant's name, tone, and the
boundaries of what it should attempt.

Common mistakes this module guards against:
  - Treating the prompt as a security boundary. It isn't one. Anything
    that actually has to hold (who can see what, who can do what) belongs
    in the RBAC/guardrail layers, not in prose here.
  - Putting secrets in the prompt. This module refuses to let you.
  - Shipping prompt changes untested. SystemPrompt is versioned and
    hashed so a change is visible and can be run through the eval suite
    (layer7_observability.evals) before it ships.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


# A short list of patterns that should never appear in a system prompt.
# This is a guard-rail for the *author* of the prompt, not a guard against
# user input (that is layer 5's job).
_SECRET_LIKE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{10,}"),          # OpenAI-style API keys
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*\S+"),
    re.compile(r"(?i)password\s*[:=]\s*\S+"),
    re.compile(r"(?i)secret\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


class SecretInPromptError(ValueError):
    """Raised when a system prompt looks like it contains a credential."""


@dataclass
class SystemPrompt:
    """
    A versioned system prompt.

    Version it like code: every change gets a new `version` string, and
    `content_hash` lets the eval suite detect "this prompt changed" even
    if nobody remembered to bump the version.
    """

    name: str
    tone: str
    rules: list[str] = field(default_factory=list)
    version: str = "v1"

    def __post_init__(self) -> None:
        self._check_for_secrets(self.name)
        self._check_for_secrets(self.tone)
        for rule in self.rules:
            self._check_for_secrets(rule)

    @staticmethod
    def _check_for_secrets(text: str) -> None:
        for pattern in _SECRET_LIKE_PATTERNS:
            if pattern.search(text):
                raise SecretInPromptError(
                    "Refusing to build a system prompt that looks like it "
                    "contains a credential. Secrets belong in environment "
                    "variables or a secrets manager, never in prompt text "
                    "-- assume anything in the prompt will eventually leak."
                )

    def render(self) -> str:
        """Render the final instruction block sent before every conversation."""
        lines = [
            f"You are {self.name}, the internal assistant.",
            f"Tone: {self.tone}.",
        ]
        lines.extend(self.rules)
        return "\n".join(lines)

    @property
    def content_hash(self) -> str:
        """Short hash of the rendered prompt, for change detection in evals."""
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "content_hash": self.content_hash,
            "rendered": self.render(),
        }


# The example straight out of the architecture write-up this project
# implements: name + tone + a couple of house rules.
DEFAULT_SYSTEM_PROMPT = SystemPrompt(
    name="Company AI",
    tone="friendly, short",
    rules=[
        "Answer only from the provided context.",
        "If you don't know, say so and point to the right team.",
        (
            "Address every distinct request in a multi-part message, using "
            "separate clearly labeled sections. Do not silently omit a request."
        ),
    ],
    version="v1",
)
