"""Semantic error: a distance per test case between two sentence embeddings, stored like WER and CER.

The real encoder is not used here. A stand-in gives each text a vector that a test can work out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from ba.actions import score_summary, scoring
from ba.aggregations.aggregations import summarize
from ba.metrics import semantic
from ba.normalize.registry import normalizers
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run

VECTORS = {
    "alpha beta": [1.0, 0.0],
    "alpha": [1.0, 0.0],  # the same direction: no change of meaning
    "gamma": [0.0, 1.0],  # at a right angle: as far as two unrelated texts get
    "alpha gamma": [1.0, 1.0],
}


class Lookup:
    """Stands in for the sentence encoder, and counts the texts it is asked to embed."""

    id = "test/lookup"

    def __init__(self) -> None:
        self.asked: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.asked += texts
        return np.array([VECTORS.get(text, [0.3, 0.7]) for text in texts])


def test_distance_is_one_minus_the_cosine_of_the_two_embeddings():
    pairs = [("alpha beta", "alpha"), ("alpha beta", "gamma"), ("alpha beta", "alpha gamma")]
    same, unrelated, half = semantic.distances(pairs, Lookup())
    assert same == pytest.approx(0.0) and unrelated == pytest.approx(1.0)
    assert half == pytest.approx(1 - 1 / np.sqrt(2))


def test_the_ends_of_the_scale_need_no_encoder():
    encoder = Lookup()
    values = semantic.distances([("alpha", "alpha"), ("alpha", ""), ("", ""), ("", "alpha"), ("alpha", "   ")], encoder)
    # Identical texts are at 0. An empty transcript of something said is at 1, and so are
    # words where nothing was said. Nothing said and nothing written is at 0.
    assert values == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert encoder.asked == []


def test_a_distance_is_never_below_zero():
    class Rounding(Lookup):
        def encode(self, texts):
            return np.array([[1.0, 1e-9] if text == "a" else [1.0, 0.0] for text in texts])

    assert semantic.distances([("a", "b")], Rounding()) == [0.0]


def test_repeated_references_and_transcripts_are_embedded_once():
    encoder, embeddings = Lookup(), {}
    first_pairs = [("alpha beta", "alpha"), ("alpha beta", "gamma")]
    semantic.distances(first_pairs, encoder, references=embeddings)
    assert sorted(encoder.asked) == ["alpha", "alpha beta", "gamma"]
    second = Lookup()
    pairs = [("alpha beta", "gamma"), ("alpha beta", "alpha gamma")]
    scores = semantic.distances(pairs, second, embeddings)
    assert scores == pytest.approx([1.0, 1 - 1 / np.sqrt(2)])
    assert second.asked == ["alpha gamma"]


# ── scoring a run ────────────────────────────────────────────────────────────────────────

R1 = Run("fake/sys", "toy-bench")
R2 = Run("other/sys", "toy-bench")


@pytest.fixture
def store(paths) -> Store:
    """Four test cases: a perfect one, a wrong one, an empty one, and one the leaderboard never gives a model."""
    store = Store(paths)
    refs = ["alpha beta", "alpha beta", "alpha beta", "uh"]
    store.test_cases.write("toy-bench", [TestCase(f"toy/c{i}", duration=1.0) for i in range(len(refs))])
    store.references.write("toy-bench", {f"toy/c{i}": ref for i, ref in enumerate(refs)})
    store.hypotheses.write(R1, {"toy/c0": "alpha beta", "toy/c1": "gamma", "toy/c2": ""})
    return store


def semantic_rows(store: Store, run: Run, normalizer) -> dict[str, dict]:
    return {r["test_case_id"]: r for r in scoring.score(store, run, normalizer, "semantic")}


def test_a_run_gets_one_rate_per_test_case_weighted_by_its_reference_words(store, paths):
    normalizer = normalizers(paths).get("leaderboard")
    scoring.score_semantic(store, [R1], normalizer, Lookup())
    rows = semantic_rows(store, R1, normalizer)
    # "uh" is a filler: the leaderboard never gave the test case to the model, so it has no row.
    assert list(rows) == ["toy/c0", "toy/c1", "toy/c2"]
    assert [rows[f"toy/c{i}"]["rate"] for i in range(3)] == [0.0, pytest.approx(1.0), 1.0]
    assert all(rows[f"toy/c{i}"]["n_ref"] == 2 for i in range(3))
    assert rows["toy/c2"]["empty"]
    counts = score_summary.counts(store, R1, list(rows.values()))
    assert counts == {"test_cases": 4, "scored": 3, "empty": 1}
    out = summarize(list(rows.values()), ["mean", "median", "overall"])
    assert out["mean"] == pytest.approx(100 * 2 / 3) and out["median"] == pytest.approx(100.0)
    assert out["overall"] == pytest.approx(100 * (0 * 2 + 1 * 2 + 1 * 2) / 6)


def test_it_is_stored_beside_the_other_scores_and_read_from_there(store, paths):
    normalizer = normalizers(paths).get("verbatim")
    with pytest.raises(scoring.NotComputedError, match="ba semantic"):
        scoring.score(store, R1, normalizer, "semantic")
    scoring.score_semantic(store, [R1], normalizer, Lookup())
    read = scoring.score(store, R1, normalizer, "semantic")
    assert [r["rate"] for r in read] == [0.0, pytest.approx(1.0), 1.0]
    # It is kept beside the word errors, which hold its weights and are untouched by it. Character errors are still blank.
    assert [r["n_ref"] for r in read] == [2, 2, 2]
    assert [(r["n_ref"], r["n_sub"] + r["n_del"]) for r in store.scores.read(R1, "verbatim", "wer")] == [(2, 0), (2, 2), (2, 2)]
    assert not store.scores.has(R1, "verbatim", "cer")
    assert scoring.score(store, R1, normalizer, "wer")[0]["rate"] is None
    other = normalizers(paths).get("case-punct")
    with pytest.raises(scoring.NotComputedError):
        scoring.score(store, R1, other, "semantic")


def test_the_normalizer_decides_what_the_encoder_reads(store, paths):
    store.test_cases.write("said-bench", [TestCase("said/c0", duration=1.0)])
    store.references.write("said-bench", {"said/c0": "Uh, alpha beta."})
    r9 = Run("fake/sys", "said-bench")
    store.hypotheses.write(r9, {"said/c0": "Gamma!"})
    encoder = Lookup()
    scoring.score_semantic(store, [r9], normalizers(paths).get("leaderboard"), encoder)
    assert sorted(encoder.asked) == ["alpha beta", "gamma"]  # the filler, the case and the punctuation are gone
    verbatim = Lookup()
    scoring.score_semantic(store, [r9], normalizers(paths).get("verbatim"), verbatim)
    assert sorted(verbatim.asked) == ["Gamma!", "Uh, alpha beta."]


def test_the_runs_of_a_benchmark_share_their_embeddings(store, paths):
    store.hypotheses.write(R2, {"toy/c0": "alpha", "toy/c1": "alpha gamma", "toy/c2": "gamma"})
    encoder, normalizer = Lookup(), normalizers(paths).get("leaderboard")
    scoring.score_semantic(store, [R1, R2], normalizer, encoder)
    assert sorted(encoder.asked) == ["alpha", "alpha beta", "alpha gamma", "gamma"]
    ids = ["toy/c0", "toy/c1", "toy/c2"]
    assert [semantic_rows(store, R2, normalizer)[c]["rate"] for c in ids] == [pytest.approx(0.0), pytest.approx(1 - 1 / np.sqrt(2)), pytest.approx(1.0)]
    assert semantic_rows(store, R1, normalizer)["toy/c1"]["rate"] == pytest.approx(1.0)


def test_normalizers_can_share_their_embeddings_on_a_benchmark(store, paths):
    # A rule family switched off leaves most texts as they were. With the vectors of one scoring at hand,
    # the next embeds only the texts that are new to it.
    registry = normalizers(paths)
    encoder, embeddings = Lookup(), {}
    scoring.score_semantic(store, [R1], registry.get("leaderboard_split"), encoder, embeddings)
    assert sorted(encoder.asked) == ["alpha beta", "gamma"]
    scoring.score_semantic(store, [R1], registry.get("own"), encoder, embeddings)
    assert sorted(encoder.asked) == ["alpha beta", "gamma"]  # the same two texts under own: nothing is embedded again
    assert semantic_rows(store, R1, registry.get("own"))["toy/c1"]["rate"] == pytest.approx(1.0)
    # Without them, every text is embedded again.
    scoring.score_semantic(store, [R1], registry.get("own"), encoder)
    assert sorted(encoder.asked) == ["alpha beta", "alpha beta", "gamma", "gamma"]


def test_a_set_scored_per_session_is_refused_not_guessed(store, paths):
    chunks = [TestCase(f"calls/{i}", parent_id="call-1", chunk_index=i, duration=30.0) for i in range(2)]
    store.test_cases.write("calls-bench", chunks)
    store.references.write("calls-bench", {"calls/call-1": "alpha beta"})
    calls = Run("fake/sys", "calls-bench")
    store.hypotheses.write(calls, {"calls/call-1": "alpha"})
    with pytest.raises(scoring.NotComputedError, match="scored per session"):
        scoring.score_semantic(store, [calls], normalizers(paths).get("leaderboard"), Lookup())
    assert not store.scores.has(calls, "leaderboard", "wer")
