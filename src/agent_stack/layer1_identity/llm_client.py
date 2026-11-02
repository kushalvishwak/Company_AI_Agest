"""
LLM client abstraction used across all layers.

Two implementations are provided behind the same `LLMClient` interface:

  - `MockLLM`   -- a small, deterministic, rule-based stand-in for a real
                   model. It needs no API key and no network access, so
                   the whole stack (including the demo and the eval
                   suite) runs out of the box. It is NOT a language model;
                   it is just enough pattern-matching to exercise every
                   layer (grounding, tool calls, refusals) predictably.
  - `OpenAICompatibleLLM` -- a thin HTTP client for any OpenAI-compatible
                   chat-completions endpoint. `GroqLLM` specializes this
                   client for the Groq API.

Swap one for the other without touching any other layer.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_call: ToolCall | None = None
    model: str = "mock-llm"
    usage_tokens: int = 0


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        context: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Produce one completion, optionally grounded in `context` and
        optionally allowed to emit a `tool_call` chosen from `tools`
        (each tool dict has at least a "name")."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock implementation: deterministic, offline, good enough to demonstrate
# every layer without requiring an API key.
# ---------------------------------------------------------------------------

_LEAVE_PATTERN = re.compile(
    r"(?i)\bleave\b.*?(\d{1,2}\s*(?:st|nd|rd|th)?\s*(?:[-–to]+\s*)?\d{0,2}\s*"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)"
)
_DATE_RANGE_PATTERN = re.compile(
    r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:[-–]|to)\s*(\d{1,2})\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*",
    re.IGNORECASE,
)


class MockLLM(LLMClient):
    """
    Rule-based stand-in for a real model. Behavior mirrors the examples
    from the architecture write-up this project implements:

      - A message that looks like a leave request, when an `apply_leave`
        tool is offered, produces a tool call instead of prose -- this is
        what "from chatbot to agent" looks like in Layer 3.
      - A question answered with retrieved `context` is grounded in that
        context and cites it -- Layer 2.
      - A question with no matching context and no tool to call gets an
        honest "I don't know" that points to a team, per the system
        prompt's own house rule, instead of a guess -- the whole point of
        Layer 1's rules and Layer 2's grounding.
    """

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        context: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        tool_names = {t["name"] for t in (tools or [])}

        # Tool-calling path (Layer 3).
        if "apply_leave" in tool_names and _LEAVE_PATTERN.search(user_message):
            match = _DATE_RANGE_PATTERN.search(user_message)
            if match:
                start_day, end_day, month = match.groups()
                date_from = f"{start_day} {month.title()}"
                date_to = f"{end_day} {month.title()}"
            else:
                # Couldn't parse a clean range; hand back what we saw
                # rather than guess at dates -- a real model should do
                # the same and ask a follow-up instead of inventing dates.
                date_from = date_to = "UNSPECIFIED"
            return LLMResponse(
                tool_call=ToolCall(
                    name="apply_leave",
                    arguments={"from": date_from, "to": date_to},
                ),
                usage_tokens=len(user_message.split()) + 8,
            )

        # Grounded-answer path (Layer 2).
        if context:
            snippet = _best_matching_sentence(user_message, context)
            if snippet:
                return LLMResponse(
                    content=f"{snippet} (source: provided context)",
                    usage_tokens=len(user_message.split()) + len(context.split()) // 4,
                )

        # Honest refusal, per the system prompt's own rule, rather than a
        # hallucinated guess -- this is the failure RAG exists to prevent.
        return LLMResponse(
            content=(
                "I don't have that in the material I was given, so I don't "
                "want to guess. Please check with the right team."
            ),
            usage_tokens=len(user_message.split()) + 6,
        )


# Common words excluded from the overlap count below, so two sentences
# that only share "the", "a", "is", "to" don't register as a match --
# without this, almost any two English sentences look related, and the
# mock "grounds" answers in context that isn't actually relevant.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "get", "gets", "how", "i", "in", "is", "it", "many",
    "my", "of", "on", "or", "please", "rate", "that", "the", "this",
    "to", "what", "when", "where", "who", "will", "with", "you", "your",
}

# Below this many shared, meaningful words, a sentence is not considered
# a real match -- it just happened to share a stopword or two.
_MIN_OVERLAP = 2


def _best_matching_sentence(question: str, context: str) -> str | None:
    """Small keyword-overlap heuristic. A real model would reason over
    the retrieved chunks; this mock just finds the sentence in `context`
    that shares the most non-stopword words with the question, so the
    demo and tests have something deterministic to assert on -- and so
    it refuses rather than grounds an answer in a merely-coincidental
    overlap. Retrieved chunks arrive as "[source] chunk text", one per
    blank-line-separated block (see layer2_rag.retriever); the "[source]"
    tag and any leading markdown heading are stripped before sentence
    splitting so they can't get glued onto the first real sentence."""
    question_words = {
        w.lower() for w in re.findall(r"[a-zA-Z]+", question)
    } - _STOPWORDS
    best_sentence, best_score = None, 0
    for block in context.split("\n\n"):
        block = re.sub(r"^\[[^\]]*\]\s*", "", block)  # strip "[source] " tag
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            words = {
                w.lower() for w in re.findall(r"[a-zA-Z]+", sentence)
            } - _STOPWORDS
            score = len(words & question_words)
            if score > best_score:
                best_sentence, best_score = sentence.strip(), score
    return best_sentence if best_score >= _MIN_OVERLAP else None


# ---------------------------------------------------------------------------
# Real implementation: any OpenAI-compatible /chat/completions endpoint.
# ---------------------------------------------------------------------------

class OpenAICompatibleLLM(LLMClient):
    """
    Minimal HTTP client for an OpenAI-compatible chat-completions API.
    Works with OpenAI-compatible chat-completions endpoints by pointing
    `base_url` at the service. `GroqLLM` supplies Groq's endpoint and
    environment-variable defaults.

    Requires the `requests` package and an API key. Reads
    `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` from the
    environment if not passed explicitly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAICompatibleLLM requires an API key. Set OPENAI_API_KEY "
                "or pass api_key explicitly. (Tip: use MockLLM for local "
                "development and tests -- no key needed.)"
            )
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or
                          "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
        self.timeout_s = timeout_s

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        context: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        import requests  # local import: optional dependency, only needed here

        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.append(
                {"role": "system", "content": f"Context:\n{context}"}
            )
        messages.append({"role": "user", "content": user_message})

        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        ),
                    },
                }
                for t in tools
            ]

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {}).get("total_tokens", 0)

        tool_calls = choice.get("tool_calls")
        if tool_calls:
            fn = tool_calls[0]["function"]
            return LLMResponse(
                tool_call=ToolCall(
                    name=fn["name"], arguments=json.loads(fn["arguments"])
                ),
                model=self.model,
                usage_tokens=usage,
            )
        return LLMResponse(
            content=choice.get("content"), model=self.model, usage_tokens=usage
        )


class GroqLLM(OpenAICompatibleLLM):
    """Chat completions using Groq's OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GroqLLM requires an API key. Set GROQ_API_KEY or enter it "
                "when launching the desktop app."
            )
        super().__init__(
            api_key=key,
            base_url=(
                base_url
                or os.environ.get("GROQ_BASE_URL")
                or "https://api.groq.com/openai/v1"
            ),
            model=model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            timeout_s=timeout_s,
        )
