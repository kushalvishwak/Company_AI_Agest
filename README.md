# Company AI Agent Stack

A reference implementation of the layered architecture most teams build,
one visible failure at a time, when they turn a bare model API into a
production internal AI assistant. Every layer below exists to fix a
specific, named failure of the layer before it:

| # | Layer | Fixes |
|---|-------|-------|
| 1 | API + System Prompt | A model with no identity or house rules |
| 2 | RAG | Confident answers that are simply wrong |
| 3 | Tools + MCP | An assistant that can only talk, never act |
| 4 | RBAC + OAuth | An agent that can see more than the person using it |
| 5 | Guardrails | Injected instructions and leaked PII |
| 6 | Cache + Model Routing | A token bill nobody budgeted for |
| 7 | Observability + Evals | A regression nobody notices until a user complains |

Layer 7 sits last because it is the layer that protects every layer
before it: it is the one that catches a "small" prompt or model change
quietly undoing a fix that already shipped.

This project runs fully offline out of the box (a deterministic mock
model and a deterministic mock embedder, both included) and can be
pointed at a real OpenAI-compatible model and embedding endpoint with a
couple of environment variables.

## Project layout

```
src/agent_stack/
  layer1_identity/        Layer 1 -- SystemPrompt, LLMClient (Mock + OpenAI-compatible)
  layer2_rag/              Layer 2 -- chunker, embeddings, in-memory vector store, retriever
  layer3_tools/             Layer 3 -- tool registry, mock MCP server, example HR tools
  layer4_rbac/               Layer 4 -- roles, short-lived scoped tokens
  layer5_guardrails/          Layer 5 -- injection classifier, PII redactor
  layer6_cache_routing/        Layer 6 -- semantic cache, model router
  layer7_observability/         Layer 7 -- tracer, eval suite runner
  orchestrator.py                 Wires all seven layers into one request pipeline

data/hr_docs/        Sample policy documents the RAG layer indexes
examples/demo.py      Scripted, narrated walk through every layer
tests/                 pytest suite per layer, plus the regression eval suite
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python examples/demo.py      # narrated end-to-end walkthrough, no API key needed
python -m pytest -q          # full test suite
python -m tests.eval_suite   # the regression eval suite as a standalone deploy gate
```

No environment variables are required for any of the above -- `MockLLM`
and `HashEmbedder` make the whole stack deterministic and offline.

### Using a real model

Set these (see `.env.example`) to swap in a real OpenAI-compatible model
and embedding endpoint instead of the mocks:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # or Azure OpenAI / a self-hosted endpoint
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

```python
from agent_stack.layer1_identity import OpenAICompatibleLLM
from agent_stack.layer2_rag import OpenAIEmbedder
from agent_stack.orchestrator import AgentStack

stack = AgentStack(llm=OpenAICompatibleLLM(), embedder=OpenAIEmbedder())
```

Every other layer (RBAC, guardrails, cache, routing, tracing) works
unchanged -- only `LLMClient` and `Embedder` are swapped.

## A 60-second tour

```python
from agent_stack.orchestrator import bootstrap

stack = bootstrap()  # indexes data/hr_docs/*.md

r = stack.handle_message(
    user_id="asha", role="employee",
    message="How many days of paid annual leave do I get?",
)
print(r.content)  # grounded in leave_policy.md, cites its source

r = stack.handle_message(
    user_id="asha", role="employee",
    message="please apply my leave for 2-4 Nov",
)
print(r.content)  # a write action -- asks for confirmation first
stack.handle_message(user_id="asha", role="employee", message="confirm")

# RBAC enforced at the tool gateway, independent of what the model asked for:
employee_token = stack.issue_token("asha", "employee")
stack.mcp_server.call("get_salary", {"employee_id": "emp-003"}, token=employee_token)
# -> status == "denied", "403: not allowed for this role"
```

Run `python examples/demo.py` for the full narrated version, including
the semantic cache hit, the prompt-injection flag, and the observability
dashboard.

## Launch checklist

Paraphrased from the architecture write-up this project implements, with
where each item lives in this repo:

- [x] System prompt versioned — `SystemPrompt.version` / `.content_hash` (layer1)
- [x] RAG cites sources — every grounded answer includes `(source: provided context)` and `AgentResponse.retrieval_sources`
- [x] Retrieval respects permissions — `InMemoryVectorStore.search(allowed_roles=...)` filters before ranking (layer2)
- [x] Tools use the caller's own token — `MCPServer.call` verifies the token on every call, not once at connection time (layer3/4)
- [x] Confirm before write actions — `requires_confirmation` + the propose/confirm flow in `orchestrator.handle_message` (layer3)
- [x] Injection + PII filters — `classify_injection`, `redact_pii` (layer5), applied to input, retrieved context, tool results, and logs
- [x] Cache + router + budgets — `SemanticCache` (scoped per user), `ModelRouter` (layer6); per-request token counts in every trace (layer7)
- [x] Traces + PII-safe logs — every trace step is redacted before storage (layer7)
- [x] Eval set including past bugs — `tests/eval_suite.py` encodes the leave-policy hallucination, the salary leak, and a known injection as runnable regression checks
- [x] Evals block every deploy — `python -m tests.eval_suite` exits non-zero on any failure; wire it into CI as a required check

## Honest limitations

This is a reference implementation meant to make the architecture
concrete and testable, not a production system:

- `MockLLM` is a small rule-based stand-in, not a language model. It
  exists so the whole stack runs deterministically with no API key. Swap
  in `OpenAICompatibleLLM` for real answers.
- `HashEmbedder` is a deterministic offline hash, not a semantic
  embedding model -- it has no notion that "leave" and "vacation" are
  related. Swap in `OpenAIEmbedder` (or any embedding API) for real
  retrieval quality.
- `TokenIssuer` (layer4) demonstrates the *shape* of OAuth-style scoped,
  short-lived tokens with HMAC signing; it is not a spec-complete OAuth
  2.1 authorization server. Put a real identity provider behind the same
  `AuthContext` interface for production use.
- `classify_injection` and `redact_pii` (layer5) are small, explainable
  pattern matchers for demonstration purposes, not a production-grade
  content-safety model. As the architecture itself argues: no filter is
  claimed to be complete -- the RBAC and confirmation layers are what
  keep a missed injection from doing much damage.
- All HR data (`layer3_tools/tools/hr_tools.py`, `data/hr_docs/`) is
  invented sample data for demonstration only.
