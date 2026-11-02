"""
The orchestrator: wires all seven layers into one request pipeline.

Mirrors the end-to-end flow the architecture write-up this project
implements describes: a system-prompted model, grounded by RAG, able to
call tools through an RBAC/OAuth-checked MCP gateway, defended by input
and output guardrails, sped up and made cheaper by a semantic cache and
model router, and traced end to end for observability.

This module intentionally contains no new ideas of its own -- it only
sequences the layers that already exist in `layerN_*`. Read it top to
bottom as the "shape of the problem" diagram turned into code.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from .layer1_identity import DEFAULT_SYSTEM_PROMPT, LLMClient, MockLLM, SystemPrompt
from .layer2_rag import HashEmbedder, Retriever
from .layer2_rag.embeddings import Embedder
from .layer3_tools import MCPServer
from .layer3_tools.tools import build_hr_tool_registry
from .layer4_rbac import ROLE_RANK, AuthContext, TokenIssuer
from .layer5_guardrails import classify_injection, redact_pii
from .layer6_cache_routing import ModelRouter, SemanticCache
from .layer7_observability import Tracer


def roles_at_or_below(role: str) -> frozenset[str]:
    """Roles whose documents/tools a caller of `role` should be able to
    see, under the simple hierarchical model in layer4_rbac.roles. An
    'hr'-only document is visible to hr and admin; an 'employee'
    document is visible to everyone."""
    caller_rank = ROLE_RANK.get(role, -1)
    return frozenset(r for r, rank in ROLE_RANK.items() if rank <= caller_rank)


@dataclass
class AgentResponse:
    content: str
    trace_id: str
    routed_model: str
    cache_hit: bool
    retrieval_sources: list[str] = field(default_factory=list)
    tool_call: dict | None = None
    tool_result_status: str | None = None
    guardrail_flagged: bool = False


class AgentStack:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        system_prompt: SystemPrompt = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm or MockLLM()
        self.embedder = embedder or HashEmbedder()
        self.system_prompt = system_prompt

        self.retriever = Retriever(self.embedder)
        self.tool_registry = build_hr_tool_registry()
        self.token_issuer = TokenIssuer()
        self.mcp_server = MCPServer(self.tool_registry, self.token_issuer)
        self.cache = SemanticCache(self.embedder)
        self.router = ModelRouter()
        self.tracer = Tracer()

        # user_id -> pending (tool_name, arguments) awaiting confirmation.
        self._pending_confirmations: dict[str, tuple[str, dict]] = {}

    # -- setup -------------------------------------------------------

    def index_document(
        self, source: str, full_text: str, *, allowed_roles: set[str]
    ) -> int:
        return self.retriever.index_document(
            source, full_text, allowed_roles=frozenset(allowed_roles)
        )

    def issue_token(self, user_id: str, role: str, ttl_seconds: int = 300) -> str:
        return self.token_issuer.issue_token(
            user_id, role, scopes=["chat"], ttl_seconds=ttl_seconds
        )

    # -- the request pipeline -----------------------------------------

    def handle_message(
        self,
        *,
        user_id: str,
        role: str,
        message: str,
        token: str | None = None,
        document_source: str | None = None,
        document_sources: list[str] | None = None,
        full_document_sources: list[str] | None = None,
    ) -> AgentResponse:
        trace = self.tracer.start_trace(user_id)
        token = token or self.issue_token(user_id, role)
        trace.record("auth.token_issued_or_provided", role=role)
        selected_sources = list(dict.fromkeys(document_sources or []))
        if document_source and document_source not in selected_sources:
            selected_sources.append(document_source)
        if document_source and document_sources is None:
            full_document_sources = list(full_document_sources or [])
            if re.search(
                r"\b(?:summarize|summarise|summary|overview)\b", message, re.I
            ):
                full_document_sources.append(document_source)
        full_sources = set(full_document_sources or [])
        cache_scope = (
            f"{user_id}:{'|'.join(selected_sources)}"
            if selected_sources
            else user_id
        )

        # A plain "confirm"/"yes" continues a pending write action
        # instead of starting a new turn. This is the two-step
        # propose-then-confirm flow the write-up calls for on write
        # actions.
        if message.strip().lower() in {"yes", "confirm", "confirmed"}:
            pending = self._pending_confirmations.pop(user_id, None)
            if pending is not None:
                tool_name, arguments = pending
                result = self.mcp_server.call(
                    tool_name, arguments, token=token, confirmed=True
                )
                trace.record("tool.confirmed_call", tool=tool_name, status=result.status)
                trace.finish()
                content = _tool_result_to_text(tool_name, result)
                return AgentResponse(
                    content=content,
                    trace_id=trace.trace_id,
                    routed_model="n/a (confirmation step)",
                    cache_hit=False,
                    tool_call={"name": tool_name, "arguments": arguments},
                    tool_result_status=result.status,
                )

        # Layer 5 (input half): flag likely injection. We do not hard
        # block on this alone -- a keyword classifier is not claimed to
        # be complete -- but we skip the cache for flagged messages so a
        # suspicious message can never poison a shared cache entry, and
        # we record it for the dashboard.
        verdict = classify_injection(message)
        if verdict.is_suspicious:
            trace.record("guardrail.injection_flagged", rules=verdict.matched_rules)

        # Layer 6 (cache half): skip on flagged input.
        cache_hit = None
        if not verdict.is_suspicious:
            cache_hit = self.cache.lookup(message, user_scope=cache_scope)
        if cache_hit is not None:
            trace.record("cache.hit", similarity=cache_hit.similarity)
            trace.finish()
            return AgentResponse(
                content=cache_hit.answer,
                trace_id=trace.trace_id,
                routed_model="n/a (cache hit)",
                cache_hit=True,
                retrieval_sources=cache_hit.sources,
                guardrail_flagged=verdict.is_suspicious,
            )
        trace.record("cache.miss")

        # Layer 6 (routing half): decide which model tier this question
        # needs. Recorded for the dashboard; see README for how to wire
        # this into a real multi-model LLMClient.
        routing = self.router.route(message)
        trace.record("router.decision", model=routing.model, reason=routing.reason)

        # Layer 2: retrieve context the caller's role is allowed to see.
        accessible_roles = roles_at_or_below(role)
        is_summary_request = bool(
            re.search(r"\b(?:summarize|summarise|summary|overview)\b", message, re.I)
        )
        if selected_sources:
            retrievals = [
                self.retriever.retrieve(
                    message,
                    allowed_roles=accessible_roles,
                    source=source,
                    include_all_source=(
                        source in full_sources
                        or (
                            source == document_source
                            and document_sources is None
                            and is_summary_request
                        )
                    ),
                )
                for source in selected_sources
            ]
            context = "\n\n".join(
                result.context for result in retrievals if result.context
            )
            chunks = [chunk for result in retrievals for chunk in result.chunks]
            stale_sources = list(
                dict.fromkeys(
                    source
                    for result in retrievals
                    for source in result.stale_sources
                )
            )
        else:
            retrieval = self.retriever.retrieve(
                message,
                allowed_roles=accessible_roles,
            )
            context = retrieval.context
            chunks = retrieval.chunks
            stale_sources = retrieval.stale_sources

        retrieval_sources = list(dict.fromkeys(c.source for c in chunks))
        if stale_sources:
            trace.record("rag.stale_index", sources=stale_sources)
        trace.record("rag.retrieved", sources=retrieval_sources)

        # Layer 1 + Layer 3: ask the model, offering it the tools this
        # role is allowed to even attempt (final enforcement still
        # happens server-side in Layer 4 on the actual call).
        tools = self.tool_registry.as_llm_tools()
        response = self.llm.complete(
            system_prompt=self.system_prompt.render(),
            user_message=message,
            context=context or None,
            tools=tools,
        )
        trace.total_tokens += response.usage_tokens

        if response.tool_call is not None:
            result = self.mcp_server.call(
                response.tool_call.name,
                response.tool_call.arguments,
                token=token,
                confirmed=False,
            )
            trace.record(
                "tool.call", tool=response.tool_call.name, status=result.status
            )
            if result.status == "error":
                trace.tool_error = True
            if result.status == "needs_confirmation":
                self._pending_confirmations[user_id] = (
                    response.tool_call.name,
                    response.tool_call.arguments,
                )
            content = _tool_result_to_text(response.tool_call.name, result)
            trace.finish()
            return AgentResponse(
                content=content,
                trace_id=trace.trace_id,
                routed_model=routing.model,
                cache_hit=False,
                retrieval_sources=retrieval_sources,
                tool_call={
                    "name": response.tool_call.name,
                    "arguments": response.tool_call.arguments,
                },
                tool_result_status=result.status,
                guardrail_flagged=verdict.is_suspicious,
            )

        # Plain text answer path.
        content = redact_pii(response.content or "")
        is_refusal = "don't have that" in content.lower()
        trace.refused = is_refusal
        trace.finish()

        # Cache the answer for next time -- never for a flagged message,
        # and scoped to this user so permission-dependent answers never
        # cross users.
        if not verdict.is_suspicious:
            self.cache.put(
                message,
                content,
                user_scope=cache_scope,
                sources=retrieval_sources,
            )

        return AgentResponse(
            content=content,
            trace_id=trace.trace_id,
            routed_model=routing.model,
            cache_hit=False,
            retrieval_sources=retrieval_sources,
            guardrail_flagged=verdict.is_suspicious,
        )


def _tool_result_to_text(tool_name: str, result) -> str:
    if result.status == "ok":
        return f"Done -- {tool_name} succeeded: {result.data}"
    if result.status == "needs_confirmation":
        return f"{result.message} Reply 'confirm' to proceed."
    if result.status == "denied":
        return "That action isn't available for your role."
    return "Something went wrong running that action."


def bootstrap(
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> AgentStack:
    """Build a fully wired AgentStack with the mock (offline) LLM and
    embedder, and the sample HR documents from data/hr_docs indexed.
    This is what the demo script and the tests use."""
    import pathlib

    stack = AgentStack(llm=llm, embedder=embedder)
    if getattr(sys, "frozen", False):
        data_dir = pathlib.Path(sys._MEIPASS) / "data" / "hr_docs"
    else:
        data_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "hr_docs"
    for doc_path in sorted(data_dir.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        # Chunking joins text back into single-line chunks (see
        # layer2_rag.chunker), so a leading "# Title" markdown heading
        # would otherwise run straight into the first sentence with no
        # punctuation between them. Dropped here, at the source, rather
        # than guessed back out of an already-flattened chunk later.
        text = re.sub(r"^#[^\n]*\n+", "", text, count=1)
        stack.index_document(doc_path.stem, text, allowed_roles={"employee"})
    return stack
