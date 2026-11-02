"""
Layer 6: Caching and Model Routing -- the cache half.

Most questions to an internal assistant are repeats. Semantic caching
embeds each incoming question and, if it is close enough in meaning to a
question already answered, returns the cached answer without a new
model call.

Two things this module is explicit about, per the common mistakes the
architecture write-up this project implements names:

  - The similarity threshold is tunable and matters: too loose, and
    unrelated questions start getting the wrong cached answer; too
    tight, and the cache barely saves anything.
  - Answers that are user-specific or permission-dependent must never be
    cached across users -- `user_scope` partitions the cache so a cached
    answer for one user is never served to another.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..layer2_rag.embeddings import Embedder, cosine_similarity


@dataclass
class CacheEntry:
    query: str
    query_vector: list[float]
    answer: str
    user_scope: str | None
    cached_at: float
    sources: list[str]


@dataclass
class CacheHit:
    answer: str
    similarity: float
    original_query: str
    sources: list[str]


class SemanticCache:
    def __init__(self, embedder: Embedder, similarity_threshold: float = 0.92) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self._entries: list[CacheEntry] = []

    def lookup(self, query: str, *, user_scope: str | None = None) -> CacheHit | None:
        query_vector = self.embedder.embed(query)
        best_entry, best_score = None, 0.0
        for entry in self._entries:
            # Never cross user/permission scopes, even for a near-perfect
            # match -- a cached answer about one user's own leave balance
            # must never be served to a different user.
            if entry.user_scope != user_scope:
                continue
            score = cosine_similarity(query_vector, entry.query_vector)
            if score > best_score:
                best_entry, best_score = entry, score

        if best_entry is not None and best_score >= self.similarity_threshold:
            return CacheHit(
                answer=best_entry.answer,
                similarity=best_score,
                original_query=best_entry.query,
                sources=list(best_entry.sources),
            )
        return None

    def put(
        self,
        query: str,
        answer: str,
        *,
        user_scope: str | None = None,
        sources: list[str] | None = None,
    ) -> None:
        self._entries.append(
            CacheEntry(
                query=query,
                query_vector=self.embedder.embed(query),
                answer=answer,
                user_scope=user_scope,
                cached_at=time.time(),
                sources=list(sources or []),
            )
        )

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
