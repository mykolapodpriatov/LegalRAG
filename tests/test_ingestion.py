"""Tests for :mod:`src.ingestion`.

These run with only ``llama-index-core`` installed -- no torch, transformers or
Streamlit -- exercising the directory-handling edge cases and chunk overlap.
"""

import pytest
from llama_index.core import Document

from src.ingestion import chunk_documents, load_documents


def test_missing_dir_is_created_and_returns_empty(tmp_path) -> None:
    """A non-existent path is created and yields an empty corpus."""
    target = tmp_path / "does_not_exist"

    docs = load_documents(str(target))

    assert docs == []
    assert target.is_dir()


def test_existing_empty_dir_returns_empty(tmp_path) -> None:
    """An existing but empty directory is a valid empty corpus, not an error."""
    assert load_documents(str(tmp_path)) == []


def test_single_txt_returns_one_document(tmp_path) -> None:
    """A directory holding one .txt file loads exactly one document."""
    (tmp_path / "contract.txt").write_text(
        "This lease agreement binds the tenant and the landlord.",
        encoding="utf-8",
    )

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert "lease agreement" in docs[0].text


def test_path_is_a_file_raises_not_a_directory(tmp_path) -> None:
    """Pointing data_dir at a file raises a clear NotADirectoryError."""
    file_path = tmp_path / "corpus.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_documents(str(file_path))


def test_chunk_documents_splits_long_doc_with_overlap() -> None:
    """A long document yields multiple chunks with real overlap between them."""
    sentences = [
        f"Clause number {i} states the party shall comply with regulation {i} in full detail."
        for i in range(120)
    ]
    text = " ".join(sentences)

    nodes = chunk_documents([Document(text=text)])

    # A long document must be split into more than one chunk.
    assert len(nodes) > 1

    # chunk_overlap=50 duplicates tokens across boundaries, so the chunk texts
    # together are strictly longer than the source document.
    assert sum(len(node.text) for node in nodes) > len(text)

    # Every consecutive chunk pair must literally share sentences (the overlap
    # window), confirming overlap rather than a clean, non-overlapping split.
    for earlier, later in zip(nodes, nodes[1:]):
        shared = [s for s in sentences if s in earlier.text and s in later.text]
        assert shared, "expected overlapping sentences between consecutive chunks"
