"""Offline retrieval-quality metrics.

Whether the passage that answers a question came back in the top-K is a fact you
can check against a labelled set. It needs no LLM judge, no key and no network,
which is what makes it something CI can actually run, unlike the faithfulness
and answer-relevance metrics on the roadmap.

Deliberately free of LlamaIndex and of any embedding model: it takes ranked ids
in and returns numbers, so the arithmetic can be tested against hand-computed
values rather than against whatever a model happened to do that day.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "QueryResult",
    "RetrievalReport",
    "evaluate_rankings",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
]


@dataclass(frozen=True)
class QueryResult:
    """One evaluated query.

    Attributes:
        query: The query text, so a regression names the query that broke
            rather than only moving a mean.
        relevant_ids: The passage ids a human says answer it.
        retrieved_ids: What the retriever actually returned, best first.
        recall: ``{k: recall@k}`` for the evaluated cut-offs.
        reciprocal_rank: ``1 / (rank + 1)`` of the first relevant hit, or 0.
    """

    query: str
    relevant_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    recall: Mapping[int, float]
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate quality over a labelled set."""

    results: tuple[QueryResult, ...]
    recall: Mapping[int, float]
    mrr: float

    @property
    def failures(self) -> tuple[QueryResult, ...]:
        """Queries whose relevant passage was never retrieved at all."""
        return tuple(r for r in self.results if r.reciprocal_rank == 0.0)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the relevant ids that appear in the first ``k`` results.

    Raises:
        ValueError: If ``k`` is below 1.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    relevant = set(relevant_ids)
    if not relevant:
        # No ground truth means nothing to recall. Returning 1.0 would let an
        # unlabelled row quietly inflate the mean.
        return 0.0
    found = sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in relevant)
    return found / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """``1 / (rank + 1)`` of the first relevant hit, or 0 when there is none.

    This is what distinguishes "found it at rank 1" from "found it at rank 9",
    which recall@10 cannot: both score 1.0 there, and only one of them is a
    retriever you would ship.
    """
    relevant = set(relevant_ids)
    for position, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant:
            return 1.0 / (position + 1)
    return 0.0


def mean_reciprocal_rank(results: Iterable[QueryResult]) -> float:
    """Mean of the per-query reciprocal ranks. 0.0 over an empty set."""
    ranks = [result.reciprocal_rank for result in results]
    return sum(ranks) / len(ranks) if ranks else 0.0


def evaluate_rankings(
    rankings: Sequence[tuple[str, Sequence[str], Sequence[str]]],
    cutoffs: Sequence[int] = (1, 3, 5),
) -> RetrievalReport:
    """Score ``(query, relevant_ids, retrieved_ids)`` rows.

    Args:
        rankings: One row per query, ``retrieved_ids`` ordered best first.
        cutoffs: The ``k`` values to report recall at.

    Returns:
        A :class:`RetrievalReport` with per-query rows and the aggregate.

    Raises:
        ValueError: If ``cutoffs`` is empty or contains a value below 1.
    """
    if not cutoffs:
        raise ValueError("at least one cutoff is required")

    results: list[QueryResult] = []
    for query, relevant, retrieved in rankings:
        results.append(
            QueryResult(
                query=query,
                relevant_ids=tuple(relevant),
                retrieved_ids=tuple(retrieved),
                recall={k: recall_at_k(retrieved, relevant, k) for k in cutoffs},
                reciprocal_rank=reciprocal_rank(retrieved, relevant),
            )
        )

    means = {
        k: (sum(r.recall[k] for r in results) / len(results) if results else 0.0) for k in cutoffs
    }
    return RetrievalReport(
        results=tuple(results),
        recall=means,
        mrr=mean_reciprocal_rank(results),
    )
