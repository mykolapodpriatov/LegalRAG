"""Tests for :mod:`src.retrieval`.

These tests deliberately avoid the heavyweight HuggingFace embedding model
(torch/transformers) and Streamlit. They inject a ``MockEmbedding`` into
``build_index`` and run against the same in-memory Qdrant backend the app uses,
proving the module is importable and usable in a minimal environment.
"""

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from src.retrieval import build_index

EMBED_DIM = 32


def _make_nodes() -> list[TextNode]:
    return [
        TextNode(text="The tenant must give thirty days notice before vacating."),
        TextNode(text="Force majeure clauses excuse performance during disasters."),
        TextNode(text="Arbitration is the agreed forum for contract disputes."),
    ]


def test_import_has_no_heavy_deps() -> None:
    """Importing the module must not pull in torch/transformers or Streamlit."""
    import sys

    import src.retrieval  # noqa: F401  (import side effect is the assertion)

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "streamlit" not in sys.modules


def test_build_index_with_injected_mock_embedding() -> None:
    """An index builds over hand-made nodes using an injected MockEmbedding."""
    embed_model = MockEmbedding(embed_dim=EMBED_DIM)

    index = build_index(_make_nodes(), embed_model=embed_model)

    assert index is not None


def test_retriever_returns_a_node() -> None:
    """A retriever over the built index returns at least one node."""
    embed_model = MockEmbedding(embed_dim=EMBED_DIM)

    index = build_index(_make_nodes(), embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=2)
    results = retriever.retrieve("What notice must a tenant give?")

    assert len(results) >= 1
    assert results[0].node.text
