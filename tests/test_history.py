"""Tests for :mod:`src.history`.

``push_history`` is a pure list helper — no Streamlit, llama-index, or
network — so the eviction contract can be checked in isolation.
"""

from src.history import DEFAULT_HISTORY_CAP, push_history


def test_push_history_appends_without_mutating_input() -> None:
    """A new item is appended and the original sequence is left unchanged."""
    history = [{"query": "first"}]
    item = {"query": "second"}

    result = push_history(history, item, cap=20)

    assert result == [{"query": "first"}, {"query": "second"}]
    assert history == [{"query": "first"}]


def test_push_history_evicts_oldest_past_cap() -> None:
    """Once past the cap, the oldest item is dropped and the newest is kept."""
    history = [{"query": n} for n in range(20)]

    result = push_history(history, {"query": 20}, cap=20)

    assert len(result) == 20
    assert result[0] == {"query": 1}
    assert result[-1] == {"query": 20}


def test_push_history_default_cap_is_twenty() -> None:
    """The default cap matches the in-session bound of 20."""
    history = [{"n": n} for n in range(DEFAULT_HISTORY_CAP)]

    result = push_history(history, {"n": DEFAULT_HISTORY_CAP})

    assert len(result) == DEFAULT_HISTORY_CAP
    assert result[0]["n"] == 1
    assert result[-1]["n"] == DEFAULT_HISTORY_CAP


def test_push_history_under_cap_keeps_all() -> None:
    """A short history is returned in full, including the new item."""
    history = [{"query": "a"}, {"query": "b"}]

    result = push_history(history, {"query": "c"}, cap=20)

    assert result == [{"query": "a"}, {"query": "b"}, {"query": "c"}]
