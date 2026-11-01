"""
Step 4-5 of the RAG pipeline: at question time, embed the question, fetch
the most similar chunks (filtered by the asker's permissions), re-rank
them, and hand back context the model is asked to answer only from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunker import chunk_text
from .embeddings import Embedder
from .vector_store import Chunk, InMemoryVectorStore


@dataclass
class RetrievalResult:
    context: str
    chunks: list[Chunk]
    stale_sources: list[str]


class Retriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.store = InMemoryVectorStore(embedder)
        self._full_texts: dict[str, str] = {}

    def index_document(
        self,
        source: str,
        full_text: str,
        *,
        allowed_roles: frozenset[str] | set[str] = frozenset({"employee"}),
    ) -> int:
        self._full_texts[source] = full_text
        chunks = chunk_text(full_text)
        ids = self.store.add_document(
            source, full_text, chunks, allowed_roles=allowed_roles
        )
        return len(ids)

    def stale_sources(self) -> list[str]:
        """Sources whose vectors no longer match their current text --
        i.e. the document changed and the index needs rebuilding."""
        return [
            source
            for source, text in self._full_texts.items()
            if self.store.check_staleness(source, text)
        ]

    def retrieve(
        self,
        query: str,
        *,
        allowed_roles: frozenset[str] | set[str],
        top_k: int = 3,
        fetch_k: int = 8,
    ) -> RetrievalResult:
        query_vector = self.embedder.embed(query)
        # Over-fetch, then re-rank with a lightweight keyword-overlap
        # pass, since similarity search alone can surface a chunk that
        # is semantically close but doesn't actually contain the words
        # the question is asking about.
        candidates = self.store.search(
            query_vector, top_k=fetch_k, allowed_roles=allowed_roles
        )
        reranked = _rerank(query, candidates)[:top_k]

        chunks = [c for c, _score in reranked]
        context = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
        return RetrievalResult(
            context=context, chunks=chunks, stale_sources=self.stale_sources()
        )


def _rerank(
    query: str, candidates: list[tuple[Chunk, float]]
) -> list[tuple[Chunk, float]]:
    query_words = set(re.findall(r"[a-zA-Z]+", query.lower()))

    def combined_score(pair: tuple[Chunk, float]) -> float:
        chunk, similarity = pair
        chunk_words = set(re.findall(r"[a-zA-Z]+", chunk.text.lower()))
        overlap = len(query_words & chunk_words) / max(len(query_words), 1)
        # Similarity gets most of the weight; keyword overlap breaks ties
        # and corrects for the mock embedder's limited semantics.
        return 0.7 * similarity + 0.3 * overlap

    return sorted(candidates, key=combined_score, reverse=True)
