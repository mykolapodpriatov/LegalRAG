"""Tests for :mod:`src.formatting`.

These use duck-typed ``types.SimpleNamespace`` fakes shaped like LlamaIndex's
``NodeWithScore`` (a ``score`` plus a ``node`` with ``text``). No llama-index,
torch, transformers or Streamlit import is required, so the citation-rendering
logic is exercised in complete isolation.
"""

from types import SimpleNamespace

from src.formatting import format_source_node, format_source_nodes


def _fake(text: str, score: float | None) -> SimpleNamespace:
    """Build a NodeWithScore-shaped fake exposing ``score`` and ``node.text``."""
    return SimpleNamespace(score=score, node=SimpleNamespace(text=text))


def test_none_score_renders_placeholder_not_literal_none() -> None:
    """A missing score must not leak the string ``"None"`` into the output."""
    result = format_source_node(_fake("A short clause.", None))

    assert "**Score:** N/A" in result
    assert "None" not in result


def test_numeric_score_is_rounded() -> None:
    """A float score is rendered compactly with fixed precision."""
    result = format_source_node(_fake("Some text.", 0.876543))

    assert "**Score:** 0.877" in result


def test_short_text_has_no_dangling_ellipsis() -> None:
    """Passages at or under the limit are shown verbatim, without an ellipsis."""
    result = format_source_node(_fake("Brief passage.", 0.5), max_chars=200)

    assert result.endswith("Brief passage.")
    assert "..." not in result


def test_long_text_is_truncated_with_ellipsis() -> None:
    """Passages longer than the limit are trimmed and gain a trailing ellipsis."""
    long_text = "x" * 500

    result = format_source_node(_fake(long_text, 0.5), max_chars=200)

    assert result.endswith("...")
    # 200 chars of preview + the 3-char ellipsis.
    assert "x" * 200 + "..." in result
    assert "x" * 201 not in result


def test_empty_text_renders_placeholder() -> None:
    """Empty or whitespace-only text yields a placeholder, not a bare label."""
    result = format_source_node(_fake("   ", 0.5))

    assert "**Text:** (no text)" in result
    assert not result.endswith("**Text:** ")


def test_format_source_nodes_returns_one_string_per_node() -> None:
    """The batch helper preserves order and length across multiple sources."""
    nodes = [
        _fake("First source.", 0.9),
        _fake("Second source.", None),
        _fake("", 0.1),
    ]

    results = format_source_nodes(nodes)

    assert len(results) == len(nodes)
    assert all(isinstance(item, str) for item in results)
    assert "First source." in results[0]
    assert "N/A" in results[1]
    assert "(no text)" in results[2]


def test_format_source_nodes_on_empty_iterable() -> None:
    """An empty iterable of sources yields an empty list."""
    assert format_source_nodes([]) == []
