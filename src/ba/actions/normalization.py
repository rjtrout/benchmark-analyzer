"""The texts as a normalizer leaves them: normalized once, and kept in the store.

A benchmark's references are normalized for every model on it at once, since they share
them. A model's transcripts are its own.
"""

from __future__ import annotations

from ba.normalize.normalizer import Normalizer
from ba.store.store import Store
from ba.store.transcriptions import Run


def normalized_references(store: Store, benchmark_id: str, normalizer: Normalizer) -> dict[str, str]:
    """A benchmark's references under a normalizer: read if they are kept, else normalized now and kept."""
    if store.references.has(benchmark_id, normalizer.id):
        return store.references.read(benchmark_id, normalizer.id)
    normalized = {}
    for test_case_id, text in store.references.read(benchmark_id).items():
        normalized[test_case_id] = normalizer(text)
    store.references.write(benchmark_id, normalized, normalizer.id)
    return normalized


def normalized_hypotheses(store: Store, run: Run, normalizer: Normalizer) -> dict[str, str]:
    """A run's transcripts under a normalizer: read if they are kept, else normalized now and kept."""
    if store.hypotheses.has(run, normalizer.id):
        return store.hypotheses.read(run, normalizer.id)
    normalized = {}
    for test_case_id, text in store.hypotheses.read(run).items():
        normalized[test_case_id] = ""
        if text:
            normalized[test_case_id] = normalizer(text)
    store.hypotheses.write(run, normalized, normalizer.id)
    return normalized


def start_again(store: Store, runs: list[Run], normalizers: list[Normalizer]) -> None:
    """Forget the texts these normalizers left of these runs, so that scoring them again starts from the texts as written.

    It is how a change to a normalizer's rules reaches the scores. A benchmark's references
    are forgotten with its runs' transcripts, and are normalized again once for all of them.
    """
    for run in runs:
        for normalizer in normalizers:
            store.references.forget(run.benchmark_id, normalizer.id)
            store.hypotheses.forget(run, normalizer.id)
