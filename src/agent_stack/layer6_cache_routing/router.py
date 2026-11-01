"""
Layer 6: Caching and Model Routing -- the routing half.

A router sends straightforward questions to a smaller, cheaper model and
reserves the large model for questions that actually need it. Published
research on learned routers has pointed to meaningful cost savings at
close to the same answer quality, though actual savings depend heavily
on traffic mix.

`ModelRouter` here is the "simple rules engine" variant the write-up
names as one of the two common approaches (the other being a small
trained classifier) -- intentionally simple rather than a tuned,
production routing model, since this project has no traffic data to
train one on.

As with the system prompt in Layer 1: a routing change is a
model-behavior change. Run it through the eval suite (layer7) before it
ships, same as any other change to what model answers what.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SMALL_MODEL = "small"
LARGE_MODEL = "large"

# Heuristics for "this needs the bigger model": long messages, multiple
# questions, or language suggesting multi-step reasoning. All tunable;
# none of this is claimed to be optimal, only a reasonable starting rule
# set for a demo router.
_COMPLEX_CUES = re.compile(
    r"(?i)\b(compare|analyze|explain why|step by step|calculate|"
    r"summarize|draft|multiple|several)\b"
)


@dataclass
class RoutingDecision:
    model: str
    reason: str


class ModelRouter:
    def __init__(
        self,
        *,
        long_message_word_threshold: int = 40,
        multi_question_threshold: int = 2,
    ) -> None:
        self.long_message_word_threshold = long_message_word_threshold
        self.multi_question_threshold = multi_question_threshold

    def route(self, message: str) -> RoutingDecision:
        word_count = len(message.split())
        question_count = message.count("?")

        if word_count > self.long_message_word_threshold:
            return RoutingDecision(
                model=LARGE_MODEL,
                reason=f"message is {word_count} words (> "
                f"{self.long_message_word_threshold})",
            )
        if question_count >= self.multi_question_threshold:
            return RoutingDecision(
                model=LARGE_MODEL,
                reason=f"message bundles {question_count} questions",
            )
        if _COMPLEX_CUES.search(message):
            match = _COMPLEX_CUES.search(message)
            return RoutingDecision(
                model=LARGE_MODEL,
                reason=f"complexity cue matched: {match.group(0)!r}",
            )
        return RoutingDecision(model=SMALL_MODEL, reason="short, simple message")
