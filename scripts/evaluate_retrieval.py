"""Measure retrieval quality against a labelled set, offline.

Changing the embedding model, the chunk size or ``similarity_top_k`` currently
produces no number at all, so there is no way to know whether a change helped.
``demo_retrieval`` prints passages for a human to eyeball, which is a smoke test
rather than a measurement.

This runs the retrieval half of the stack over a labelled ``(query,
relevant_ids)`` set and reports recall@k and MRR. No Anthropic key, no ~2 GB
model, no torch, exactly like the demo.

**What it does and does not tell you.** Retrieval here runs under
:class:`LexicalEmbedding`, a deterministic bag-of-words embedder defined below,
because LlamaIndex's ``MockEmbedding`` returns the same constant vector for
every text and would make the ranking arbitrary. So these numbers are not a
verdict on the real multilingual-E5 model's semantics. They are a regression
gate on everything around it: chunking, ``similarity_top_k``, the index
configuration, the retrieval wiring. Point ``--embed-model real`` at the actual
model when you want the other question answered.

Run it from the repository root::

    python -m scripts.evaluate_retrieval
    python -m scripts.evaluate_retrieval --fail-under recall@5=0.8
    python -m scripts.evaluate_retrieval --json > baseline.json
    python -m scripts.evaluate_retrieval --baseline baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode

# Make ``src`` importable when run directly as a script rather than as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import RetrievalReport, evaluate_rankings  # noqa: E402
from src.retrieval import build_index  # noqa: E402

EMBED_DIM = 64
DEFAULT_CUTOFFS = (1, 3, 5)
DEFAULT_TOP_K = 5

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: The corpus, keyed by id so a labelled set can point at passages.
CORPUS: dict[str, str] = {
    "notice": "The tenant must give thirty days written notice before vacating the premises.",
    "force-majeure": (
        "Force majeure clauses excuse a party's performance during natural disasters or war."
    ),
    "arbitration": (
        "Arbitration seated in London is the agreed forum for resolving contract disputes."
    ),
    "termination": (
        "Either party may terminate this agreement for a material breach left uncured "
        "for fourteen days."
    ),
    "confidentiality": (
        "Confidential information disclosed under this agreement must not be shared "
        "with third parties."
    ),
    "deposit": (
        "The landlord shall return the security deposit within thirty days of the end "
        "of the tenancy, less any deductions for damage."
    ),
    "rent-increase": (
        "Rent may be increased once per calendar year with sixty days written notice to the tenant."
    ),
    "governing-law": "This agreement is governed by the laws of England and Wales.",
    "assignment": (
        "Neither party may assign its rights under this agreement without the prior "
        "written consent of the other."
    ),
    "liability-cap": (
        "The total liability of either party is capped at the fees paid in the twelve "
        "months preceding the claim."
    ),
    "notices-service": (
        "Notices under this agreement are served by email to the addresses listed in "
        "Schedule 1 and are deemed received on the next business day."
    ),
    "ip-ownership": (
        "All intellectual property created in the course of the services vests in the "
        "customer on payment in full."
    ),
}

#: The labelled set: what a human says answers each query.
LABELLED_QUERIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("What written notice must a tenant give before vacating?", ("notice",)),
    ("How long does the landlord have to return the security deposit?", ("deposit",)),
    ("Can the rent be increased, and with how much notice?", ("rent-increase",)),
    ("Where are contract disputes resolved?", ("arbitration",)),
    ("Which law governs this agreement?", ("governing-law",)),
    ("What excuses performance during a natural disaster or war?", ("force-majeure",)),
    ("How long does a party have to cure a material breach?", ("termination",)),
    ("May a party assign its rights without consent?", ("assignment",)),
    ("What is the cap on total liability?", ("liability-cap",)),
    ("How are notices served, and when are they deemed received?", ("notices-service",)),
    ("Who owns intellectual property created during the services?", ("ip-ownership",)),
    (
        "What are the rules about confidential information and third parties?",
        ("confidentiality",),
    ),
)


def _hashed_bag_of_words(text: str, dim: int) -> list[float]:
    """A deterministic, L2-normalised bag-of-words vector.

    Tokens are hashed into ``dim`` buckets and counted, then normalised, so two
    texts sharing vocabulary score higher under cosine similarity than unrelated
    ones. Crude, but it reacts to the text, which is the one thing
    ``MockEmbedding`` does not do.
    """
    vector = [0.0] * dim
    for token in _TOKEN_RE.findall(text.casefold()):
        bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % dim
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        # An empty or punctuation-only passage has no direction; leaving it at
        # the origin keeps it out of every ranking rather than tying with
        # everything.
        return vector
    return [value / norm for value in vector]


def _make_embedding_model(kind: str) -> Any:
    """Build the embedder for a run.

    ``lexical`` is the offline default. ``real`` loads the multilingual-E5 model
    the application uses, which needs the heavy dependencies and a download, and
    is only worth it when the question is about the model rather than about the
    pipeline around it.
    """
    if kind == "real":
        from src.retrieval import get_embed_model

        return get_embed_model()

    from llama_index.core.embeddings import BaseEmbedding

    class LexicalEmbedding(BaseEmbedding):
        """Deterministic bag-of-words embedder. Same input, same numbers, always."""

        def _get_query_embedding(self, query: str) -> list[float]:
            return _hashed_bag_of_words(query, EMBED_DIM)

        def _get_text_embedding(self, text: str) -> list[float]:
            return _hashed_bag_of_words(text, EMBED_DIM)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    return LexicalEmbedding(embed_dim=EMBED_DIM)


#: Metadata key carrying the labelled id. Not the node id: Qdrant requires a
#: UUID or an integer there, so a human-readable key cannot live in it.
ID_KEY = "chunk_id"


def _nodes() -> list[TextNode]:
    """The corpus as LlamaIndex nodes, each carrying its labelled id.

    The id is excluded from the embedded text. Left in, the id token would be
    part of what gets embedded, and a query mentioning "deposit" would match the
    deposit passage partly because of its name rather than its content.
    """
    return [
        TextNode(
            text=text,
            metadata={ID_KEY: chunk_id},
            excluded_embed_metadata_keys=[ID_KEY],
            excluded_llm_metadata_keys=[ID_KEY],
        )
        for chunk_id, text in CORPUS.items()
    ]


def run_evaluation(
    *,
    top_k: int = DEFAULT_TOP_K,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    embed_model: str = "lexical",
) -> RetrievalReport:
    """Retrieve for every labelled query and score the rankings."""
    index = build_index(_nodes(), embed_model=_make_embedding_model(embed_model))
    retriever = index.as_retriever(similarity_top_k=top_k)

    rankings = []
    for query, relevant in LABELLED_QUERIES:
        retrieved = [
            str(node.node.metadata.get(ID_KEY, node.node.node_id))
            for node in retriever.retrieve(query)
        ]
        rankings.append((query, relevant, retrieved))
    return evaluate_rankings(rankings, cutoffs=cutoffs)


def report_to_dict(report: RetrievalReport) -> dict[str, Any]:
    """A stable, sorted JSON payload, suitable for saving as a baseline."""
    return {
        "recall": {f"recall@{k}": value for k, value in sorted(report.recall.items())},
        "mrr": report.mrr,
        "queries": [
            {
                "query": result.query,
                "relevant_ids": list(result.relevant_ids),
                "retrieved_ids": list(result.retrieved_ids),
                "recall": {f"recall@{k}": v for k, v in sorted(result.recall.items())},
                "reciprocal_rank": result.reciprocal_rank,
            }
            for result in report.results
        ],
    }


def parse_threshold(spec: str) -> tuple[str, float]:
    """Parse ``metric=value``, e.g. ``recall@5=0.8`` or ``mrr=0.6``.

    Raises:
        ValueError: On anything else. A malformed threshold must be a usage
            error, not a check that silently does not run.
    """
    metric, _, raw = spec.partition("=")
    metric = metric.strip()
    if not metric or not raw.strip():
        raise ValueError(f"expected metric=value, got {spec!r}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{spec!r}: {raw.strip()!r} is not a number") from exc
    return metric, value


def metric_value(report: RetrievalReport, metric: str) -> float:
    """Look a metric up by its reported name.

    Raises:
        KeyError: If the name is not one this report carries.
    """
    if metric == "mrr":
        return report.mrr
    if metric.startswith("recall@"):
        try:
            k = int(metric.removeprefix("recall@"))
        except ValueError as exc:
            raise KeyError(metric) from exc
        if k in report.recall:
            return report.recall[k]
    raise KeyError(metric)


def _print_report(report: RetrievalReport) -> None:
    for k, value in sorted(report.recall.items()):
        print(f"recall@{k}: {value:.3f}")
    print(f"mrr: {report.mrr:.3f}")
    if report.failures:
        # Which query broke, not only that the mean moved.
        print(f"\n{len(report.failures)} query(s) retrieved nothing relevant:")
        for result in report.failures:
            print(f"  - {result.query}")


def _compare(report: RetrievalReport, baseline_path: Path) -> int:
    """Print the delta against a saved run. Returns the exit code."""
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    previous_recall = payload.get("recall", {})
    worse = False
    print("\nagainst baseline:")
    for k, value in sorted(report.recall.items()):
        name = f"recall@{k}"
        before = previous_recall.get(name)
        if before is None:
            print(f"  {name}: {value:.3f} (not in baseline)")
            continue
        delta = value - float(before)
        worse = worse or delta < 0
        print(f"  {name}: {before:.3f} -> {value:.3f} ({delta:+.3f})")
    before_mrr = payload.get("mrr")
    if before_mrr is not None:
        delta = report.mrr - float(before_mrr)
        worse = worse or delta < 0
        print(f"  mrr: {float(before_mrr):.3f} -> {report.mrr:.3f} ({delta:+.3f})")
    return 1 if worse else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate_retrieval",
        description="Measure retrieval quality against a labelled set, offline.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="similarity_top_k.")
    parser.add_argument(
        "--cutoffs",
        default=",".join(str(k) for k in DEFAULT_CUTOFFS),
        help="Comma-separated k values to report recall at.",
    )
    parser.add_argument(
        "--embed-model",
        choices=("lexical", "real"),
        default="lexical",
        help=(
            "lexical (default) is deterministic and offline. real loads the "
            "multilingual-E5 model, which needs the heavy dependencies."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="A saved --json report to compare against; exit 1 if any metric dropped.",
    )
    parser.add_argument(
        "--fail-under",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Exit 1 when a metric is below VALUE, e.g. recall@5=0.8. Repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    try:
        cutoffs = tuple(int(part) for part in args.cutoffs.split(",") if part.strip())
    except ValueError:
        print(
            f"--cutoffs: {args.cutoffs!r} is not a comma-separated list of integers",
            file=sys.stderr,
        )
        return 2
    if not cutoffs or any(k < 1 for k in cutoffs):
        print("--cutoffs: every value must be an integer >= 1", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print(f"--top-k must be at least 1, got {args.top_k}", file=sys.stderr)
        return 2

    thresholds = []
    for spec in args.fail_under:
        try:
            thresholds.append(parse_threshold(spec))
        except ValueError as exc:
            print(f"--fail-under: {exc}", file=sys.stderr)
            return 2

    report = run_evaluation(top_k=args.top_k, cutoffs=cutoffs, embed_model=args.embed_model)

    if args.json:
        json.dump(report_to_dict(report), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_report(report)

    exit_code = 0
    for metric, threshold in thresholds:
        try:
            value = metric_value(report, metric)
        except KeyError:
            print(f"--fail-under: unknown metric {metric!r}", file=sys.stderr)
            return 2
        if value < threshold:
            print(f"{metric} {value:.3f} is below {threshold:.3f}", file=sys.stderr)
            exit_code = 1

    if args.baseline is not None:
        try:
            exit_code = _compare(report, args.baseline) or exit_code
        except (OSError, ValueError) as exc:
            print(f"--baseline: {exc}", file=sys.stderr)
            return 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
