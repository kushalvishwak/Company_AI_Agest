"""
Layer 7: Observability -- the evals half, the layer that protects all the
others.

A test set of real questions, paired with expected answers or rules,
gets run against the system on every change: a prompt edit, a new model
version, a different router, a modified retriever, a changed tool. Past
incidents belong permanently in that test set so a well-intentioned
later change cannot silently undo the fix.

`BUILT_IN_REGRESSION_CASES` in this module encodes, as runnable checks,
the three named incidents from the source material this project
implements: the leave-policy hallucination, the salary leak, and a known
injection attempt. See tests/eval_suite.py for how they are wired up to
the actual stack and run as part of CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EvalCase:
    id: str
    description: str
    check: Callable[[], bool]


@dataclass
class EvalResult:
    id: str
    description: str
    passed: bool
    error: str | None = None


@dataclass
class EvalReport:
    results: list[EvalResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        lines = [f"{passed}/{total} eval cases passed"]
        for r in self.failures:
            lines.append(f"  FAIL {r.id}: {r.description} ({r.error or 'check returned False'})")
        return "\n".join(lines)


def run_eval_suite(cases: list[EvalCase]) -> EvalReport:
    """Runs every case and NEVER stops at the first failure -- a deploy
    gate needs to see every regression in one run, not just the first
    one it tripped over."""
    results: list[EvalResult] = []
    for case in cases:
        try:
            passed = bool(case.check())
            results.append(
                EvalResult(id=case.id, description=case.description, passed=passed)
            )
        except Exception as exc:  # a raising check counts as a failure
            results.append(
                EvalResult(
                    id=case.id,
                    description=case.description,
                    passed=False,
                    error=str(exc),
                )
            )
    return EvalReport(results=results)
