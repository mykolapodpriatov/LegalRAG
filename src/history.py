"""Pure helpers for a bounded in-session query history.

Extracted from the Streamlit UI so eviction can be unit-tested without a
Streamlit runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")

DEFAULT_HISTORY_CAP = 20


def push_history(history: Sequence[T], item: T, cap: int = DEFAULT_HISTORY_CAP) -> list[T]:
    """Append ``item`` and drop the oldest entries once past ``cap``.

    Returns a new list; ``history`` is not mutated. When ``cap`` is 0 or
    negative the result is empty.
    """
    if cap <= 0:
        return []
    updated = [*history, item]
    return updated[-cap:]
