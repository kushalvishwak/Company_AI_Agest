"""
Step 3 of the RAG pipeline: store embedded chunks so the most similar
ones can be found at question time.

This store also carries two things the architecture write-up calls out
as easy to get wrong:

  - `allowed_roles` on every chunk, so retrieval can be filtered by the
    requesting user's permissions (the same RBAC check from Layer 4,
    applied to RAG) instead of returning a document the asking user
    should never see.
  - `source_hash` per source document, so a stale index (the policy
    changed, the vectors didn't) can be detected instead of silently
    serving an out-of-date chunk. See `InMemoryVectorStore.stale_sources`.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field

from .embeddings import Embedder, cosine_similarity


@dataclass
class Chunk:
    id: str
    text: str
    vector: list[float]
    source: str
    allowed_roles: frozenset[str]
    source_hash: str
    indexed_at: float = field(default_factory=time.time)


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: dict[str, Chunk] = {}
        # Tracks the hash of each source document's full text at the time
        # it was last indexed, so we can tell a caller "this document
        # changed since you embedded it" -- the stale-index mistake.
        self._source_hashes: dict[str, str] = {}

    def add_document(
        self,
        source: str,
        full_text: str,
        chunks: list[str],
        *,
        allowed_roles: frozenset[str] | set[str] = frozenset({"employee"}),
    ) -> list[str]:
        allowed_roles = frozenset(allowed_roles)
        source_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:12]

        # Remove any previously indexed chunks for this source so we
        # never mix stale and fresh chunks for the same document.
        self.remove_source(source)

        ids: list[str] = []
        for chunk_text_ in chunks:
            vector = self.embedder.embed(chunk_text_)
            chunk_id = str(uuid.uuid4())
            self._chunks[chunk_id] = Chunk(
                id=chunk_id,
                text=chunk_text_,
                vector=vector,
                source=source,
                allowed_roles=allowed_roles,
                source_hash=source_hash,
            )
            ids.append(chunk_id)

        self._source_hashes[source] = source_hash
        return ids

    def remove_source(self, source: str) -> None:
        for chunk_id in [c.id for c in self._chunks.values() if c.source == source]:
            del self._chunks[chunk_id]

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 3,
        allowed_roles: frozenset[str] | set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks by cosine similarity, filtering first by the
        requesting user's roles. Filtering happens BEFORE ranking, not
        after, so a highly-relevant chunk the user can't see never has a
        chance to be returned -- "ignoring permissions" is exactly the
        RAG mistake this guards against."""
        candidates = self._chunks.values()
        if allowed_roles is not None:
            candidates = [
                c for c in candidates if c.allowed_roles & frozenset(allowed_roles)
            ]
        scored = [
            (c, cosine_similarity(query_vector, c.vector)) for c in candidates
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def check_staleness(self, source: str, current_full_text: str) -> bool:
        """Returns True if `current_full_text` no longer matches what was
        last indexed for `source` -- i.e. the index is stale and needs
        re-embedding before it can be trusted."""
        current_hash = hashlib.sha256(
            current_full_text.encode("utf-8")
        ).hexdigest()[:12]
        indexed_hash = self._source_hashes.get(source)
        return indexed_hash is not None and indexed_hash != current_hash

    def __len__(self) -> int:
        return len(self._chunks)
