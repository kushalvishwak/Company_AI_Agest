"""
Step 1 of the RAG pipeline: chunk documents into small, overlapping
pieces so a sentence is never cut in half at a chunk boundary.

Chunk size is a real trade-off, not a detail: chunks too big bury the
relevant sentence in noise, chunks too small lose the surrounding
context that would have made the retrieved answer correct. The defaults
here (120 words, 20-word overlap) are a reasonable starting point, not a
tuned value -- tune against your own documents and eval set.
"""

from __future__ import annotations

import re


def chunk_text(
    text: str,
    *,
    chunk_size_words: int = 120,
    overlap_words: int = 20,
) -> list[str]:
    if chunk_size_words <= overlap_words:
        raise ValueError("chunk_size_words must be greater than overlap_words")

    words = re.findall(r"\S+", text)
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size_words - overlap_words
    while start < len(words):
        piece = words[start : start + chunk_size_words]
        chunks.append(" ".join(piece))
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks
