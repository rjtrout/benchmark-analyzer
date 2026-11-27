"""A summary of scores: for a run under a normalizer, the figures and the counts a table shows beside them."""

from __future__ import annotations

from typing import Any

from ba.actions.normalization import start_again
from ba.actions.scoring import NotComputedError, score
from ba.aggregations.aggregations import summarize
from ba.normalize.normalizer import Normalizer
from ba.store.store import Store
from ba.store.transcriptions import Run


def counts(store: Store, run: Run, rows: list[dict[str, Any]]) -> dict[str, int]:
    """The counts every table shows beside a score.

    ``test_cases`` is what the benchmark has, in test cases or in sessions. ``scored`` is how
    many of them the model has a transcript of, and ``empty`` how many of those it wrote
    nothing for, which count as all deletions.
    """
    empty = 0
    for row in rows:
        if row["empty"]:
            empty += 1
    return {"test_cases": store.references.count(run.benchmark_id), "scored": len(rows), "empty": empty}


def score_table(
    store: Store,
    runs: list[Run],
    normalizers: list[Normalizer],
    aggregation_ids: list[str],
    rescore: bool = False,
    metric_ids: tuple[str, ...] | list[str] = ("wer",),
) -> list[dict[str, Any]]:
    """One row per run, normalizer and metric, with every count beside the number.

    Semantic error gets no row where it is not computed yet. ``rescore`` scores again from
    the texts as written, even where a score is saved.
    """
    if rescore:
        start_again(store, runs, normalizers)
    table = []
    for run in runs:
        which = {"run_id": run.run_id, "system": run.system_id, "benchmark": run.benchmark_id}
        for metric_id in metric_ids:
            for normalizer in normalizers:
                try:
                    rows = score(store, run, normalizer, metric_id, rescore=rescore)
                except NotComputedError:
                    continue
                reference_units = sum(row["n_ref"] for row in rows)
                figures = summarize(rows, aggregation_ids)
                table.append(
                    {
                        **which,
                        "normalizer": normalizer.id,
                        "metric": metric_id,
                        **counts(store, run, rows),
                        "ref_units": reference_units,
                        **figures,
                    }
                )
    return table
