"""Tests for :mod:`src.generation`.

These stay in the same minimal-deps tier as the rest of the suite
(``pytest`` + ``llama-index-core``): Anthropic is never imported when an
``llm`` is injected, so CI does not need ``llama-index-llms-anthropic``.
"""

from types import SimpleNamespace

import pytest
from llama_index.core.llms import MockLLM

from src.generation import generate_response, get_query_engine


class _FakeIndex:
    """Records ``as_query_engine`` kwargs and returns a canned engine."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None
        self.engine = object()

    def as_query_engine(self, **kwargs):
        self.kwargs = kwargs
        return self.engine


def test_get_query_engine_raises_when_api_key_unset(monkeypatch) -> None:
    """Missing ``ANTHROPIC_API_KEY`` is a ValueError, as production requires."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_query_engine(_FakeIndex())


def test_get_query_engine_accepts_injected_llm(monkeypatch) -> None:
    """An injected llm builds the engine with the production retrieval settings."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    index = _FakeIndex()
    llm = MockLLM()

    engine = get_query_engine(index, llm=llm)

    assert engine is index.engine
    assert index.kwargs is not None
    assert index.kwargs["llm"] is llm
    assert index.kwargs["similarity_top_k"] == 5
    assert index.kwargs["response_mode"] == "compact"


def test_generate_response_delegates_to_query_engine() -> None:
    """``generate_response`` returns whatever ``query_engine.query`` returns."""
    seen: dict[str, str] = {}
    expected = SimpleNamespace(response="the tenant must give thirty days notice")

    def _query(query_text: str):
        seen["query_text"] = query_text
        return expected

    query_engine = SimpleNamespace(query=_query)

    result = generate_response(query_engine, "What notice must a tenant give?")

    assert result is expected
    assert seen["query_text"] == "What notice must a tenant give?"
