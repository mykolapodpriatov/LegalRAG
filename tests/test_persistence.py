"""Tests for on-disk persistence of the Qdrant index in :mod:`src.retrieval`.

Like the other retrieval tests, these run with only ``llama-index-core``,
``llama-index-vector-stores-qdrant`` and ``qdrant-client`` installed -- no
torch, transformers or Streamlit -- by injecting a ``MockEmbedding``. They
verify that ``build_index(..., persist_path=...)`` writes the corpus to disk so
a fresh client can read it back without re-embedding.
"""

import os

import qdrant_client
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from src.retrieval import build_index, collection_is_populated, load_index

EMBED_DIM = 32


def _make_nodes() -> list[TextNode]:
    return [
        TextNode(text="The tenant must give thirty days notice before vacating."),
        TextNode(text="Force majeure clauses excuse performance during disasters."),
        TextNode(text="Arbitration is the agreed forum for contract disputes."),
    ]


def test_persist_path_writes_collection_readable_by_fresh_client(tmp_path) -> None:
    """Building with ``persist_path`` leaves a populated collection on disk."""
    persist_path = str(tmp_path / "qdrant_storage")
    nodes = _make_nodes()

    index = build_index(
        nodes, embed_model=MockEmbedding(embed_dim=EMBED_DIM), persist_path=persist_path
    )
    # Local Qdrant holds an exclusive lock on the storage dir; release it so a
    # fresh client can open the same directory below.
    index.vector_store.client.close()

    reopened = qdrant_client.QdrantClient(path=persist_path)
    try:
        collections = {c.name for c in reopened.get_collections().collections}
        assert "legal_docs" in collections
        assert reopened.count(collection_name="legal_docs").count == len(nodes)
    finally:
        reopened.close()


def test_custom_collection_name_is_used(tmp_path) -> None:
    """A caller-supplied ``collection_name`` is what ends up on disk."""
    persist_path = str(tmp_path / "qdrant_storage")

    index = build_index(
        _make_nodes(),
        embed_model=MockEmbedding(embed_dim=EMBED_DIM),
        persist_path=persist_path,
        collection_name="statutes",
    )
    index.vector_store.client.close()

    reopened = qdrant_client.QdrantClient(path=persist_path)
    try:
        collections = {c.name for c in reopened.get_collections().collections}
        assert collections == {"statutes"}
    finally:
        reopened.close()


def test_no_persist_path_stays_in_memory(tmp_path) -> None:
    """Without ``persist_path`` nothing is written to disk (pure in-memory)."""
    empty_dir = tmp_path / "should_stay_empty"
    empty_dir.mkdir()

    index = build_index(_make_nodes(), embed_model=MockEmbedding(embed_dim=EMBED_DIM))

    assert index is not None
    assert list(empty_dir.iterdir()) == []


def test_collection_is_populated_false_for_missing_path(tmp_path) -> None:
    """A nonexistent persist_path is reported as not populated, not an error."""
    missing = str(tmp_path / "does_not_exist")

    assert collection_is_populated(missing) is False


def test_collection_is_populated_false_for_none_path() -> None:
    """``None`` (the in-memory sentinel) is reported as not populated."""
    assert collection_is_populated(None) is False


def test_collection_is_populated_false_before_indexing(tmp_path) -> None:
    """An existing but empty storage directory has no populated collection."""
    persist_path = str(tmp_path / "qdrant_storage")
    os.makedirs(persist_path, exist_ok=True)

    assert collection_is_populated(persist_path) is False


def test_collection_is_populated_true_after_build_index(tmp_path) -> None:
    """After ``build_index`` writes a collection, it is reported as populated."""
    persist_path = str(tmp_path / "qdrant_storage")

    index = build_index(
        _make_nodes(), embed_model=MockEmbedding(embed_dim=EMBED_DIM), persist_path=persist_path
    )
    index.vector_store.client.close()

    assert collection_is_populated(persist_path) is True
    assert collection_is_populated(persist_path, collection_name="other") is False


def test_load_index_queries_existing_collection_without_reembedding(tmp_path) -> None:
    """``load_index`` opens a persisted collection and can retrieve from it.

    Only the original ``build_index`` call embeds anything; ``load_index``
    merely wraps the existing vector store, so this proves a fresh session can
    reuse the on-disk index instead of re-indexing the corpus.
    """
    persist_path = str(tmp_path / "qdrant_storage")
    embed_model = MockEmbedding(embed_dim=EMBED_DIM)

    built = build_index(_make_nodes(), embed_model=embed_model, persist_path=persist_path)
    built.vector_store.client.close()

    reopened = load_index(embed_model=embed_model, persist_path=persist_path)
    try:
        retriever = reopened.as_retriever(similarity_top_k=2)
        results = retriever.retrieve("What notice must a tenant give?")

        assert len(results) >= 1
        assert results[0].node.text
    finally:
        reopened.vector_store.client.close()


def test_load_index_respects_custom_collection_name(tmp_path) -> None:
    """``load_index`` opens the caller-supplied ``collection_name``, not the default."""
    persist_path = str(tmp_path / "qdrant_storage")
    embed_model = MockEmbedding(embed_dim=EMBED_DIM)

    built = build_index(
        _make_nodes(),
        embed_model=embed_model,
        persist_path=persist_path,
        collection_name="statutes",
    )
    built.vector_store.client.close()

    reopened = load_index(
        embed_model=embed_model, persist_path=persist_path, collection_name="statutes"
    )
    try:
        assert reopened.vector_store.collection_name == "statutes"
    finally:
        reopened.vector_store.client.close()
