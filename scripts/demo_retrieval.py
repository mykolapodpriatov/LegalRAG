"""Offline retrieval smoke-test for the LegalRAG pipeline.

Exercises the retrieval half of the stack with **no** heavyweight dependencies:
a handful of hardcoded legal ``TextNode``s are indexed with a lightweight
``MockEmbedding`` (32-dim vectors) instead of the ~2 GB multilingual-E5 model,
and no Anthropic API key is needed because answer generation is skipped. This
lets anyone run ``src.retrieval.build_index`` plus a retriever end-to-end in
well under a second.

Run it from the repository root::

    python -m scripts.demo_retrieval
    # or, equivalently:
    python scripts/demo_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

# Make ``src`` importable when this file is run directly as a script
# (``python scripts/demo_retrieval.py``) rather than as a module, since a bare
# script puts its own directory -- not the repo root -- on ``sys.path``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.formatting import format_source_nodes  # noqa: E402
from src.retrieval import build_index  # noqa: E402

EMBED_DIM = 32
DEFAULT_QUERY = "What notice must a tenant give before leaving?"

# A small, self-contained corpus so the demo needs no external data files.
SAMPLE_TEXTS = [
    "The tenant must give thirty days written notice before vacating the premises.",
    "Force majeure clauses excuse a party's performance during natural disasters or war.",
    "Arbitration seated in London is the agreed forum for resolving contract disputes.",
    "Either party may terminate this agreement for a material breach left uncured for fourteen days.",
    "Confidential information disclosed under this agreement must not be shared with third parties.",
]


def _build_nodes() -> list[TextNode]:
    """Return the hardcoded demo corpus as LlamaIndex ``TextNode``s."""
    return [TextNode(text=text) for text in SAMPLE_TEXTS]


def main(query: str = DEFAULT_QUERY, top_k: int = 3) -> list:
    """Index the demo corpus offline and print the top ``top_k`` hits.

    Args:
        query: Natural-language query to retrieve for.
        top_k: Number of top passages to retrieve and display.

    Returns:
        The retrieved ``NodeWithScore`` results (length ``>= 1`` for a non-empty
        corpus), so callers and tests can assert on them.
    """
    index = build_index(_build_nodes(), embed_model=MockEmbedding(embed_dim=EMBED_DIM))
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    print(f"Query: {query}\n")
    print(f"Top {len(results)} passage(s):\n")
    for rank, citation in enumerate(format_source_nodes(results), start=1):
        print(f"{rank}. {citation}\n")

    return results


if __name__ == "__main__":
    main()
