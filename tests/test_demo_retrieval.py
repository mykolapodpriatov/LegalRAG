"""Tests for the offline retrieval demo script.

Invokes ``scripts.demo_retrieval.main`` -- which uses a ``MockEmbedding`` and no
API key -- and asserts it retrieves passages. Runs with only llama-index-core,
llama-index-vector-stores-qdrant and qdrant-client installed; no torch,
transformers or Streamlit.
"""

from scripts.demo_retrieval import main


def test_demo_returns_at_least_one_hit() -> None:
    """The demo retrieves at least one passage from the hardcoded corpus."""
    results = main()

    assert len(results) >= 1
    assert results[0].node.text


def test_demo_respects_top_k() -> None:
    """``top_k`` caps how many passages the demo returns."""
    results = main(query="termination for material breach", top_k=2)

    assert 1 <= len(results) <= 2
