"""
Company AI Agent Stack
=======================

A reference implementation of the layered architecture most teams build,
one visible failure at a time, when they turn a bare model API into a
production internal AI assistant:

  1. API + System Prompt   -> identity and tone
  2. RAG                   -> grounded, non-hallucinated answers
  3. Tools + MCP            -> the assistant can act, not just talk
  4. RBAC + OAuth            -> the assistant only sees what the user can
  5. Guardrails              -> prompt-injection and PII defenses
  6. Cache + Routing         -> cost control
  7. Observability + Evals  -> the layer that protects all the others

See README.md for the full write-up and the launch checklist.
"""

__version__ = "0.1.0"
