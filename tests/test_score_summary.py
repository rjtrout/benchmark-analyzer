"""A summary of scores: the table of every run, normalizer and metric, with the counts beside each figure."""

from __future__ import annotations

import pytest

from ba.actions.score_summary import counts, score_table
from ba.actions.scoring import score
from ba.normalize.normalizer import Normalizer
from ba.normalize.registry import normalizers
from tests.conftest import TOY_RUN, make_run


def test_the_counts_beside_a_score_are_of_the_benchmark_and_of_the_models_transcripts(paths):
    store = make_run(paths, statuses=["ok", "empty", "skipped"])
    rows = score(store, TOY_RUN, normalizers(paths).get("leaderboard"))
    # The benchmark has three test cases. The model has a transcript of two, and wrote nothing in one of those.
    assert counts(store, TOY_RUN, rows) == {"test_cases": 3, "scored": 2, "empty": 1}


def test_the_score_table_has_a_row_per_run_normalizer_and_metric_with_its_counts(paths):
    store = make_run(paths, statuses=["ok", "empty", "skipped"])
    registry = normalizers(paths)
    table = score_table(store, [TOY_RUN], [registry.get("verbatim"), registry.get("leaderboard")], ["overall", "mean"], metric_ids=["wer", "cer", "semantic"])
    # Semantic error is computed on request: until then it has no row, and nothing fails.
    assert [(r["metric"], r["normalizer"]) for r in table] == [("wer", "verbatim"), ("wer", "leaderboard"), ("cer", "verbatim"), ("cer", "leaderboard")]
    board = table[1]
    assert (board["run_id"], board["system"], board["benchmark"]) == ("fake-sys__toy-bench", "fake/sys", "toy-bench")
    assert (board["test_cases"], board["scored"], board["empty"], board["ref_units"]) == (3, 2, 1, 5)
    assert board["overall"] == pytest.approx(100 * 3 / 5) and board["mean"] == pytest.approx(100 * (1 / 3 + 1) / 2)


def test_scoring_again_starts_from_the_texts_as_written(paths):
    store = make_run(paths)  # both transcripts are "alpha beta gama", of "alpha beta gamma" and of "delta epsilon"
    as_written = Normalizer("toy", lambda text: text)
    assert score_table(store, [TOY_RUN], [as_written], ["overall"])[0]["overall"] == pytest.approx(100 * 4 / 5)
    # The same normalizer with new rules: one changes a transcript, the other a reference.
    changed = Normalizer("toy", lambda text: text.replace("gama", "gamma").replace("delta epsilon", "alpha beta"))
    # The store is a snapshot. It keeps no record of the rules a score was made under, so a saved score is read.
    assert score_table(store, [TOY_RUN], [changed], ["overall"])[0]["overall"] == pytest.approx(100 * 4 / 5)
    # Scored again, the new rules reach both texts: one word too many in five.
    assert score_table(store, [TOY_RUN], [changed], ["overall"], rescore=True)[0]["overall"] == pytest.approx(100 * 1 / 5)
    assert store.hypotheses.read(TOY_RUN, "toy") == {"toy/c0": "alpha beta gamma", "toy/c1": "alpha beta gamma"}
    assert store.references.read("toy-bench", "toy")["toy/c1"] == "alpha beta"
