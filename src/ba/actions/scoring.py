"""Scoring a run: its metrics under one normalizer, a row for each transcript, kept in the store.

A model is scored on what it has a transcript of: test cases, or whole sessions on a set that
is scored per session. The leaderboard never gives a model a test case whose reference its
own normalizer empties, so no model has a transcript of one, and every figure here is of the
test cases the leaderboard scores.

A metric is made when it is first asked for, and kept: word errors, and character errors
when they are wanted. Semantic error is made by its own command. The store says whether a
metric is kept. The texts a metric compares are as a normalizer leaves them: see
``normalization.py``.
"""

from __future__ import annotations

from typing import Any

from ba.actions.normalization import normalized_hypotheses, normalized_references
from ba.metrics import semantic as semantic_metric
from ba.metrics.registry import METRICS
from ba.normalize.normalizer import Normalizer
from ba.store.store import Store
from ba.store.transcriptions import Run


class NotComputedError(ValueError):
    """Semantic error has not been computed for this run yet, or cannot be."""


def score(
    store: Store,
    run: Run,
    normalizer: Normalizer,
    metric_id: str = "wer",
    rescore: bool = False,
) -> list[dict[str, Any]]:
    """A run's rows for one metric, one for each transcript: read if the metric is kept, else counted and kept.

    A row is what the aggregations read. For word and character errors it holds the counts:
    ``n_ref``, ``n_sub``, ``n_del`` and ``n_ins``. For semantic error it holds the ``rate``,
    and the reference's word count in ``n_ref`` as its weight.

    ``rescore`` counts again. To score again from the texts as written, call
    ``normalization.start_again`` first. Semantic error is the exception: it needs a sentence
    encoder, so it is only ever read here, and ``ba semantic`` or the Results page computes it.
    """
    count = METRICS.get(metric_id)  # an unknown metric is an error before anything is read
    kept = store.scores.has(run, normalizer.id, metric_id)

    if metric_id == "semantic":
        if not kept:
            raise NotComputedError(
                f"Semantic error is not computed yet for '{run.run_id}' under the {normalizer.id} normalizer. "
                "Compute it on the Results page, or with: ba semantic"
            )
    elif rescore or not kept:
        references = normalized_references(store, run.benchmark_id, normalizer)
        hypotheses = normalized_hypotheses(store, run, normalizer)
        rows = []
        for test_case_id, as_written in store.hypotheses.read(run).items():
            counts = count(references[test_case_id], hypotheses[test_case_id], normalizer)
            rows.append({"test_case_id": test_case_id, "empty": not as_written.strip(), **counts})
        store.scores.write(run, normalizer.id, metric_id, rows)

    return store.scores.read(run, normalizer.id, metric_id)


def score_semantic(
    store: Store,
    runs: list[Run],
    normalizer: Normalizer,
    embedder: semantic_metric.Embedder,
    embeddings: dict[str, Any] | None = None,
) -> None:
    """Semantic error for each transcript of runs on one benchmark, kept beside their other scores.

    It is the distance between the embeddings of the reference and the transcript, both
    normalized. Its weight in the overall figure is the reference's word count, which the
    word errors beside it hold. A set scored per session is refused: a whole call is far
    more text than a sentence encoder reads.

    The runs share their embeddings, so a text is embedded once. ``embeddings`` is where
    they are kept: hand the same dictionary to the scoring of another normalizer on the
    benchmark, and the texts the two leave alike are not embedded again.
    """
    if embeddings is None:
        embeddings = {}
    for run in runs:
        if store.test_cases.scored_per_session(run.benchmark_id):
            raise NotComputedError(
                f"'{run.benchmark_id}' is scored per session, and a whole session is far more text than a sentence "
                "encoder reads. Semantic error is given for sets scored per test case."
            )
        rows = score(store, run, normalizer, "wer")  # the word errors are kept first: they hold the weights
        references = normalized_references(store, run.benchmark_id, normalizer)
        hypotheses = normalized_hypotheses(store, run, normalizer)
        pairs = []
        for row in rows:
            test_case_id = row["test_case_id"]
            pairs.append((references[test_case_id], hypotheses[test_case_id]))
        distances = semantic_metric.distances(pairs, embedder, embeddings)
        for row, distance in zip(rows, distances, strict=True):
            row["rate"] = float(distance)
        store.scores.write(run, normalizer.id, "semantic", rows)
