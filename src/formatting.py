"""Pure formatting helpers for rendering retrieval source citations.

Extracted from ``app.py`` so the citation strings can be unit-tested without a
Streamlit runtime or any heavyweight retrieval/embedding dependency. The
functions operate on duck-typed objects that expose ``score`` and ``node.text``
(as LlamaIndex's ``NodeWithScore`` does), so tests may pass simple
``types.SimpleNamespace`` fakes rather than constructing real nodes.

The inline version in ``app.py`` printed a literal ``"None"`` for missing
scores and a dangling ``"..."`` for passages shorter than the cut-off; both are
fixed here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_ELLIPSIS = "..."
_MISSING_SCORE = "N/A"
_EMPTY_TEXT = "(no text)"


def _format_score(score: Any) -> str:
    """Render a relevance score, tolerating a missing (``None``) value.

    A real ``NodeWithScore`` may carry ``score=None`` (e.g. when the retriever
    does not attach one); rendering that verbatim leaked the string ``"None"``
    into the UI. Numeric scores are rounded for a compact, stable display.
    """
    if score is None:
        return _MISSING_SCORE
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return str(score)


def _truncate(text: str, max_chars: int) -> str:
    """Trim ``text`` to ``max_chars``, appending an ellipsis only when trimmed.

    Passages at or under the limit are returned unchanged, so short snippets no
    longer gain a misleading trailing ``"..."``.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + _ELLIPSIS


def format_source_node(source_node: Any, max_chars: int = 200) -> str:
    """Format a single retrieval source into a Markdown citation string.

    Args:
        source_node: A duck-typed node exposing ``score`` and ``node.text``.
        max_chars: Maximum number of characters of passage text to show before
            truncating with an ellipsis.

    Returns:
        A Markdown snippet with the (possibly missing) score and a trimmed,
        non-empty passage preview.
    """
    score = getattr(source_node, "score", None)
    node = getattr(source_node, "node", None)
    text = getattr(node, "text", "") or ""

    preview = _truncate(text.strip(), max_chars) if text.strip() else _EMPTY_TEXT
    return f"**Score:** {_format_score(score)}\n\n**Text:** {preview}"


def format_source_nodes(
    source_nodes: Iterable[Any], max_chars: int = 200
) -> list[str]:
    """Format an iterable of retrieval sources into Markdown citation strings.

    Args:
        source_nodes: Iterable of duck-typed nodes exposing ``score`` and
            ``node.text`` (e.g. LlamaIndex ``NodeWithScore`` objects).
        max_chars: Maximum passage characters to show per source before
            truncating with an ellipsis.

    Returns:
        One formatted citation string per input node, in order.
    """
    return [format_source_node(node, max_chars) for node in source_nodes]
