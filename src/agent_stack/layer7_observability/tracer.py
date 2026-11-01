"""
Layer 7: Observability -- the tracing half.

None of the earlier layers answer the question asked after an incident:
why did it answer that way, and has this happened before? Observability
means tracing every request end to end: the prompt that went in, which
documents retrieval pulled, every tool call and its result, the model's
response, and the timing and token cost at each step.

Logs get the same PII-redaction discipline as the guardrails layer,
since a log is just another place sensitive data can leak -- every step
recorded here is redacted before it is stored.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from ..layer5_guardrails.pii_redactor import redact_pii


@dataclass
class TraceStep:
    name: str
    data: dict
    at: float


@dataclass
class Trace:
    trace_id: str
    user_id: str
    started_at: float
    steps: list[TraceStep] = field(default_factory=list)
    finished_at: float | None = None
    total_tokens: int = 0
    refused: bool = False
    tool_error: bool = False

    def record(self, name: str, **data) -> None:
        # Redact every value before it is ever stored, not just the
        # user-facing output -- a trace is a log, and logs leak too.
        safe_data = {k: redact_pii(str(v)) for k, v in data.items()}
        self.steps.append(TraceStep(name=name, data=safe_data, at=time.time()))

    def finish(self) -> None:
        self.finished_at = time.time()

    @property
    def latency_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "latency_s": self.latency_s,
            "total_tokens": self.total_tokens,
            "refused": self.refused,
            "tool_error": self.tool_error,
            "steps": [
                {"name": s.name, "data": s.data, "at": s.at} for s in self.steps
            ],
        }


class Tracer:
    """Collects traces in memory and derives the dashboard numbers the
    write-up names: cost (via total_tokens as a proxy), latency, refusal
    rate, and tool-error rate."""

    def __init__(self) -> None:
        self._traces: list[Trace] = []

    def start_trace(self, user_id: str) -> Trace:
        trace = Trace(trace_id=str(uuid.uuid4()), user_id=user_id, started_at=time.time())
        self._traces.append(trace)
        return trace

    @property
    def traces(self) -> list[Trace]:
        return list(self._traces)

    def dashboard(self) -> dict:
        finished = [t for t in self._traces if t.finished_at is not None]
        n = len(finished)
        if n == 0:
            return {
                "request_count": 0,
                "avg_latency_s": None,
                "total_tokens": 0,
                "refusal_rate": None,
                "tool_error_rate": None,
            }
        return {
            "request_count": n,
            "avg_latency_s": sum(t.latency_s or 0 for t in finished) / n,
            "total_tokens": sum(t.total_tokens for t in finished),
            "refusal_rate": sum(1 for t in finished if t.refused) / n,
            "tool_error_rate": sum(1 for t in finished if t.tool_error) / n,
        }
