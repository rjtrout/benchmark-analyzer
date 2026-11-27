"""The reproduction: our score of each model's public transcripts beside the leaderboard's own.

The leaderboard publishes a figure in two places. Its table shows a WER per model and set,
to two decimals. Beside most transcript files its scorer leaves a score file, with the WER
and the three error counts behind it. Ours is the WER under the leaderboard's normalizer, over
all the test cases it scores, rounded the way its scorer rounds.
"""

from __future__ import annotations

from typing import Any

from ba.actions import score_summary
from ba.actions.normalization import start_again
from ba.actions.scoring import score
from ba.aggregations.aggregations import overall
from ba.config import BenchmarkConfig, Config, SystemConfig
from ba.normalize.normalizer import Normalizer
from ba.store.store import Store
from ba.store.transcriptions import Run


def reproduction(store: Store, cfg: Config, normalizer: Normalizer, rescore: bool = False) -> list[dict[str, Any]]:
    """One row per model and benchmark: ours, the score file's figure, the table's, and a verdict.

    ``match``              every figure published for the pair is reproduced
    ``counts differ``      the figures are reproduced, and the score file reaches its figure
                           from other error counts than ours
    ``DIFFERS``            a published WER is not the one these transcripts give
    ``nothing published``  there is no figure to compare with
    ``not fetched``        there is no score of ours to compare

    ``rescore`` scores again from the transcripts as written, even where a score is saved.
    """
    if rescore:
        system_ids = [system.id for system in cfg.systems()]
        benchmark_ids = [benchmark.id for benchmark in cfg.benchmarks()]
        start_again(store, store.hypotheses.runs(system_ids, benchmark_ids), [normalizer])
    rows = []
    for system in cfg.systems():
        for benchmark in cfg.benchmarks():
            rows.append(_reproduce(store, system, benchmark, normalizer, rescore))
    return rows


def _reproduce(
    store: Store,
    system: SystemConfig,
    benchmark: BenchmarkConfig,
    normalizer: Normalizer,
    rescore: bool,
) -> dict[str, Any]:
    """One model on one benchmark: our WER, the two published ones, and whether they agree."""
    row: dict[str, Any] = {
        "system": system.id,
        "benchmark": benchmark.id,
        "table": system.published.get(benchmark.id),  # the WER the leaderboard's table shows
        "score_file": None,  # the WER and error counts its scorer left beside the transcript file
        "ours": None,
        "counts": None,
        "scored": None,
        "test_cases": None,
        "checks": [],  # one True or False for each published figure: is it the one we get?
        "counts_match": None,
    }
    run = Run(system.id, benchmark.id)
    if not store.hypotheses.has(run):
        row["verdict"] = "not fetched"
        return row

    kept = score(store, run, normalizer, "wer", rescore=rescore)
    counts = score_summary.counts(store, run, kept)
    ours = overall(kept)
    our_counts = {
        "ins": sum(test_case["n_ins"] for test_case in kept),
        "del": sum(test_case["n_del"] for test_case in kept),
        "sub": sum(test_case["n_sub"] for test_case in kept),
    }
    score_file = None
    if store.published_scores.has(run):
        published = store.published_scores.read(run)
        score_file = {
            "wer": published["wer"],
            "ins": published["wer_ins"],
            "del": published["wer_del"],
            "sub": published["wer_sub"],
        }
    row.update(
        ours=ours,
        counts=our_counts,
        score_file=score_file,
        scored=counts["scored"],
        test_cases=counts["test_cases"],
    )

    # The leaderboard publishes its figures to two decimals, so ours is rounded the same way.
    if score_file is not None:
        row["checks"].append(score_file["wer"] == round(ours, 2))
        row["counts_match"] = (
            score_file["ins"] == our_counts["ins"]
            and score_file["del"] == our_counts["del"]
            and score_file["sub"] == our_counts["sub"]
        )
    if row["table"] is not None:
        row["checks"].append(row["table"] == round(ours, 2))

    if not row["checks"]:
        row["verdict"] = "nothing published"
    elif not all(row["checks"]):
        row["verdict"] = "DIFFERS"
    elif row["counts_match"] is False:
        row["verdict"] = "counts differ"
    else:
        row["verdict"] = "match"
    return row
