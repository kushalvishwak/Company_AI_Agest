"""
Step 2 of the RAG pipeline: embed each chunk into a vector that captures
its meaning, so similar meaning lands near similar meaning.

Two implementations behind one `Embedder` interface:

  - `HashEmbedder`   -- a deterministic, offline bag-of-words hashing
                        embedding. Not semantically rich (it has no idea
                        "leave" and "vacation" are related), but it needs
                        no API key and no network, so the RAG pipeline,
                        the demo, and the eval suite all run out of the
                        box. Good enough to demonstrate chunking,
                        storage, retrieval, and permission filtering.
  - `OpenAIEmbedder` -- real embeddings via an OpenAI-compatible
                        `/embeddings` endpoint, used when
                        `OPENAI_API_KEY` is set.

Swap one for the other without touching the vector store or retriever.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod


class Embedder(ABC):
    dimensions: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashEmbedder(Embedder):
    """Deterministic offline embedding: hash each token into one of
    `dimensions` buckets, accumulate, then L2-normalize. Same text
    always produces the same vector; related words that happen to share
    no tokens will NOT be considered related -- that limitation is the
    price of needing no model and no network for local dev and tests."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbedder(Embedder):
    """Real embeddings via an OpenAI-compatible `/embeddings` endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAIEmbedder requires an API key. Set OPENAI_API_KEY or "
                "pass api_key explicitly. (Tip: use HashEmbedder for local "
                "development and tests -- no key needed.)"
            )
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or
                          "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import requests  # local import: optional dependency, only needed here

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [row["embedding"] for row in data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)
