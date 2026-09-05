"""Tests for the offline retrieval-quality metrics.

The arithmetic is checked against hand-computed values on fixed rankings, so a
change to the metric is caught by the metric's own test rather than by a number
moving somewhere downstream.
"""

from __future__ import annotations

import pytest

from src.evaluation import (
    evaluate_rankings,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# recall@k
# ---------------------------------------------------------------------------


def test_recall_is_one_when_the_relevant_passage_is_in_the_cut() -> None:
    assert recall_at_k(["a", "b", "c"], ["b"], 3) == 1.0


def test_recall_is_zero_when_it_falls_outside_the_cut() -> None:
    # The whole point of a cut-off: rank 3 is not in the top 2.
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0


def test_recall_is_the_fraction_when_several_passages_are_relevant() -> None:
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5


def test_recall_over_an_empty_ranking_is_zero() -> None:
    assert recall_at_k([], ["a"], 5) == 0.0


def test_recall_with_no_ground_truth_is_zero_not_one() -> None:
    # Returning 1.0 would let an unlabelled row quietly inflate the mean.
    assert recall_at_k(["a"], [], 1) == 0.0


def test_recall_rejects_a_cutoff_below_one() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k(["a"], ["a"], 0)


# ---------------------------------------------------------------------------
# reciprocal rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_is_one_at_the_top() -> None:
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_reciprocal_rank_falls_off_with_position() -> None:
    # This is what recall@10 cannot tell you: both of these have recall 1.0.
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_came_back() -> None:
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_hit() -> None:
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


def test_mrr_over_an_empty_set_is_zero() -> None:
    assert mean_reciprocal_rank([]) == 0.0


# ---------------------------------------------------------------------------
# evaluate_rankings
# ---------------------------------------------------------------------------


def _rankings() -> list[tuple[str, list[str], list[str]]]:
    return [
        ("found first", ["a"], ["a", "b", "c"]),
        ("found third", ["c"], ["a", "b", "c"]),
        ("not found", ["z"], ["a", "b", "c"]),
    ]


def test_aggregate_recall_is_the_mean_over_queries() -> None:
    report = evaluate_rankings(_rankings(), cutoffs=(1, 3))

    # recall@1: 1, 0, 0 -> 1/3.  recall@3: 1, 1, 0 -> 2/3.
    assert report.recall[1] == pytest.approx(1 / 3)
    assert report.recall[3] == pytest.approx(2 / 3)


def test_aggregate_mrr_is_the_mean_reciprocal_rank() -> None:
    report = evaluate_rankings(_rankings(), cutoffs=(3,))

    assert report.mrr == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


def test_per_query_rows_name_the_query_that_broke() -> None:
    # A drop should point at a query, not only move a mean.
    report = evaluate_rankings(_rankings(), cutoffs=(3,))

    assert [r.query for r in report.failures] == ["not found"]


def test_every_query_gets_a_row() -> None:
    report = evaluate_rankings(_rankings(), cutoffs=(1,))

    assert len(report.results) == 3
    assert report.results[0].retrieved_ids == ("a", "b", "c")


def test_an_empty_set_scores_zero_rather_than_raising() -> None:
    report = evaluate_rankings([], cutoffs=(1, 5))

    assert report.recall == {1: 0.0, 5: 0.0}
    assert report.mrr == 0.0


def test_at_least_one_cutoff_is_required() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        evaluate_rankings(_rankings(), cutoffs=())
