"""End-to-end tests for the offline retrieval evaluation.

These index the real corpus with the deterministic lexical embedder, so they
need no API key, no ~2 GB model and no torch, matching how the demo test runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_retrieval import (
    CORPUS,
    LABELLED_QUERIES,
    main,
    metric_value,
    parse_threshold,
    report_to_dict,
    run_evaluation,
)
from src.evaluation import RetrievalReport


def test_every_labelled_id_exists_in_the_corpus() -> None:
    # A typo in a label would silently score as "never retrieved" forever.
    for _query, relevant in LABELLED_QUERIES:
        for chunk_id in relevant:
            assert chunk_id in CORPUS, chunk_id


def test_the_lexical_embedder_actually_retrieves() -> None:
    """MockEmbedding returns one constant vector for every text, which would
    make the ranking arbitrary and these numbers meaningless."""
    report = run_evaluation()

    assert isinstance(report, RetrievalReport)
    assert report.recall[5] > 0.5
    assert report.mrr > 0.5


def test_two_runs_produce_identical_numbers() -> None:
    first = run_evaluation()
    second = run_evaluation()

    assert report_to_dict(first) == report_to_dict(second)


def test_a_smaller_top_k_cannot_improve_recall() -> None:
    wide = run_evaluation(top_k=5, cutoffs=(5,))
    narrow = run_evaluation(top_k=1, cutoffs=(5,))

    assert narrow.recall[5] <= wide.recall[5]


def test_the_cli_prints_the_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    out = capsys.readouterr().out
    assert "recall@1:" in out
    assert "mrr:" in out


def test_the_cli_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "recall@5" in payload["recall"]
    assert len(payload["queries"]) == len(LABELLED_QUERIES)


def test_fail_under_exits_one_below_the_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--fail-under", "recall@1=1.01"]) == 1
    assert "below" in capsys.readouterr().err


def test_fail_under_exits_zero_at_or_above_the_threshold() -> None:
    assert main(["--fail-under", "recall@5=0.0"]) == 0


@pytest.mark.parametrize("spec", ["recall@5", "recall@5=", "=0.8", "recall@5=high", ""])
def test_a_malformed_threshold_is_a_usage_error(spec: str) -> None:
    # Not a check that silently does not run.
    with pytest.raises(ValueError):
        parse_threshold(spec)


def test_an_unknown_metric_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--fail-under", "precision@5=0.8"]) == 2
    assert "unknown metric" in capsys.readouterr().err


def test_metric_lookup_covers_the_reported_names() -> None:
    report = run_evaluation(cutoffs=(1, 3))

    assert metric_value(report, "mrr") == report.mrr
    assert metric_value(report, "recall@3") == report.recall[3]
    with pytest.raises(KeyError):
        metric_value(report, "recall@99")


@pytest.mark.parametrize("args", [["--top-k", "0"], ["--cutoffs", "0"], ["--cutoffs", "a,b"]])
def test_nonsense_arguments_exit_two(args: list[str]) -> None:
    assert main(args) == 2


def test_a_baseline_comparison_flags_a_drop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline.json"
    payload = report_to_dict(run_evaluation())
    # An impossible baseline: every real run is a regression against it.
    payload["recall"]["recall@1"] = 1.0
    payload["mrr"] = 1.0
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--baseline", str(baseline)]) == 1
    assert "against baseline" in capsys.readouterr().out


def test_a_baseline_comparison_passes_when_nothing_dropped(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report_to_dict(run_evaluation())), encoding="utf-8")

    assert main(["--baseline", str(baseline)]) == 0


def test_a_missing_baseline_file_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--baseline", "no-such-file.json"]) == 2
    assert "--baseline" in capsys.readouterr().err
