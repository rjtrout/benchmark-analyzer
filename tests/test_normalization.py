"""The texts as a normalizer leaves them: normalized once, kept, and forgotten when scoring starts again."""

from __future__ import annotations

import dataclasses

from ba.actions.normalization import normalized_hypotheses, normalized_references, start_again
from ba.normalize.normalizer import Normalizer
from ba.normalize.registry import normalizers
from ba.store.transcriptions import Run
from tests.conftest import TOY_RUN, make_run


def watching(normalizer: Normalizer, seen: list[str]) -> Normalizer:
    """The same normalizer, noting every text it is handed."""

    def watched(text: str) -> str:
        seen.append(text)
        return normalizer.fn(text)

    return dataclasses.replace(normalizer, fn=watched)


def test_a_benchmarks_references_are_normalized_once_and_kept(paths):
    store = make_run(paths)
    seen = []
    normalizer = watching(normalizers(paths).get("leaderboard"), seen)
    # Every reference of the benchmark, whichever model is being scored: the models on it share them.
    left = {"toy/c0": "alpha beta gamma", "toy/c1": "delta epsilon", "toy/c2": ""}
    assert normalized_references(store, "toy-bench", normalizer) == left
    assert seen == ["alpha beta gamma", "delta epsilon", "uh"]
    assert store.references.read("toy-bench", "leaderboard") == left
    # Asked for again, they are read.
    assert normalized_references(store, "toy-bench", normalizer) == left
    assert len(seen) == 3


def test_a_runs_transcripts_are_normalized_once_and_kept(paths):
    store = make_run(paths)
    run = Run("other/sys", "toy-bench")
    store.hypotheses.write(run, {"toy/c0": "Alpha, beta!", "toy/c1": ""})
    seen = []
    normalizer = watching(normalizers(paths).get("leaderboard"), seen)
    left = {"toy/c0": "alpha beta", "toy/c1": ""}
    assert normalized_hypotheses(store, run, normalizer) == left
    assert seen == ["Alpha, beta!"]  # a transcript with nothing in it is not handed to the normalizer
    assert store.hypotheses.read(run, "leaderboard") == left
    # Asked for again, they are read.
    assert normalized_hypotheses(store, run, normalizer) == left
    assert seen == ["Alpha, beta!"]
    # They are this model's own: another model's transcripts are not normalized with them.
    assert not store.hypotheses.has(TOY_RUN, "leaderboard")


def test_starting_again_forgets_the_texts_a_normalizer_left(paths):
    store = make_run(paths)
    for normalizer_id in ("toy", "other"):
        capitals = Normalizer(normalizer_id, lambda text: text.upper())
        normalized_references(store, "toy-bench", capitals)
        normalized_hypotheses(store, TOY_RUN, capitals)
    # The same normalizer with new rules. The store keeps no record of the rules, so the kept texts are read.
    as_written = Normalizer("toy", lambda text: text)
    assert normalized_references(store, "toy-bench", as_written)["toy/c0"] == "ALPHA BETA GAMMA"
    assert normalized_hypotheses(store, TOY_RUN, as_written)["toy/c0"] == "ALPHA BETA GAMA"
    # Started again, both texts are normalized under the new rules.
    start_again(store, [TOY_RUN], [as_written])
    assert normalized_references(store, "toy-bench", as_written)["toy/c0"] == "alpha beta gamma"
    assert normalized_hypotheses(store, TOY_RUN, as_written)["toy/c0"] == "alpha beta gama"
    # Another normalizer's texts are left as they were.
    assert store.references.read("toy-bench", "other")["toy/c0"] == "ALPHA BETA GAMMA"
    assert store.hypotheses.read(TOY_RUN, "other")["toy/c0"] == "ALPHA BETA GAMA"
