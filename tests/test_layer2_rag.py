from agent_stack.layer2_rag import HashEmbedder, Retriever, chunk_text
from agent_stack.layer2_rag.embeddings import cosine_similarity


def test_chunk_text_overlaps_and_covers_all_words():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size_words=100, overlap_words=20)
    assert len(chunks) > 1
    # last word must appear in the final chunk -- nothing gets dropped
    assert "word299" in chunks[-1]
    # overlap: end of one chunk should share words with the start of the next
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words


def test_hash_embedder_is_deterministic():
    embedder = HashEmbedder()
    v1 = embedder.embed("the quick brown fox")
    v2 = embedder.embed("the quick brown fox")
    assert v1 == v2


def test_cosine_similarity_identical_text_is_high():
    embedder = HashEmbedder()
    v1 = embedder.embed("annual leave policy")
    v2 = embedder.embed("annual leave policy")
    assert cosine_similarity(v1, v2) > 0.99


def test_retriever_respects_role_permissions():
    retriever = Retriever(HashEmbedder())
    retriever.index_document(
        "general_policy",
        "All employees get 18 days of paid annual leave per year.",
        allowed_roles=frozenset({"employee"}),
    )
    retriever.index_document(
        "hr_only_policy",
        "HR-only note: executive leave approvals route through legal.",
        allowed_roles=frozenset({"hr", "admin"}),
    )

    as_employee = retriever.retrieve(
        "leave approvals", allowed_roles=frozenset({"employee"})
    )
    sources = {c.source for c in as_employee.chunks}
    assert "hr_only_policy" not in sources

    as_hr = retriever.retrieve("leave approvals", allowed_roles=frozenset({"hr"}))
    sources_hr = {c.source for c in as_hr.chunks}
    assert "hr_only_policy" in sources_hr


def test_stale_source_detection():
    retriever = Retriever(HashEmbedder())
    retriever.index_document("doc1", "Version one of the policy.", allowed_roles={"employee"})
    assert retriever.stale_sources() == []

    # Simulate the underlying document changing without a re-index.
    retriever._full_texts["doc1"] = "Version two of the policy, totally different."
    assert "doc1" in retriever.stale_sources()


def test_remove_document_clears_text_chunks_and_staleness_metadata():
    retriever = Retriever(HashEmbedder())
    retriever.index_document("uploads/guide.txt", "Company holiday schedule.", allowed_roles={"employee"})

    retriever.remove_document("uploads/guide.txt")

    assert "uploads/guide.txt" not in retriever._full_texts
    assert not any(
        chunk.source == "uploads/guide.txt"
        for chunk in retriever.store._chunks.values()
    )
    assert "uploads/guide.txt" not in retriever.stale_sources()
    assert "uploads/guide.txt" not in retriever.store._source_hashes


def test_source_scoped_summary_includes_every_chunk_from_only_that_document():
    retriever = Retriever(HashEmbedder())
    target_source = "uploads/Kushal_InfoQ_Article_Draft.docx"
    target_text = " ".join(
        [f"Article section{i} contains InfoQ research." for i in range(250)]
        + ["distinctivefinalfact"]
    )
    retriever.index_document(
        target_source, target_text, allowed_roles={"employee"}
    )
    retriever.index_document(
        "uploads/IBM_Guide.pdf",
        "IBM guide has unrelated information.",
        allowed_roles={"employee"},
    )

    result = retriever.retrieve(
        "summarize the infoQ_Article",
        allowed_roles={"employee"},
        source=target_source,
        include_all_source=True,
    )

    assert result.chunks
    assert {chunk.source for chunk in result.chunks} == {target_source}
    assert "distinctivefinalfact" in result.context
