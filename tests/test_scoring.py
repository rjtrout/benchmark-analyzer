"""Scoring a run: which test cases count, the texts it keeps, and a row per transcript for each metric."""

from __future__ import annotations

import pytest

from ba.actions import score_summary
from ba.actions.scoring import score
from ba.aggregations.aggregations import summarize
from ba.normalize.registry import normalizers
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run
from tests.conftest import TOY_RUN, make_run

# ── which test cases count ───────────────────────────────────────────────────────────────


def test_a_model_is_scored_on_the_test_cases_it_has_a_transcript_of(paths):
    store = make_run(paths, statuses=["ok", "empty", "skipped"])
    rows = score(store, TOY_RUN, normalizers(paths).get("leaderboard"))
    # "uh" has no transcript: the leaderboard never gives a model a test case whose reference its normalizer empties.
    assert [r["test_case_id"] for r in rows] == ["toy/c0", "toy/c1"]
    assert (rows[0]["n_ref"], rows[0]["n_sub"], rows[0]["empty"], rows[0]["rate"]) == (3, 1, False, None)
    # A model that was given a test case and wrote nothing is charged for every word.
    assert (rows[1]["n_ref"], rows[1]["n_del"], rows[1]["empty"]) == (2, 2, True)
    assert score_summary.counts(store, TOY_RUN, rows) == {"test_cases": 3, "scored": 2, "empty": 1}
    assert summarize(rows, ["overall"])["overall"] == pytest.approx(100 * 3 / 5)


def test_a_set_scored_per_session_has_a_row_for_each_session(paths):
    store = Store(paths)
    # Two calls, in chunks. A call has one reference, and a model's transcripts are of whole calls.
    chunks = [
        TestCase("calls/a2", parent_id="call-a", chunk_index=1),
        TestCase("calls/a1", parent_id="call-a", chunk_index=0),
        TestCase("calls/a3", parent_id="call-a", chunk_index=2),
        TestCase("calls/b1", parent_id="call-b", chunk_index=0),
    ]
    store.test_cases.write("calls-bench", chunks)
    store.references.write("calls-bench", {"calls/call-a": "alpha beta gamma delta", "calls/call-b": "epsilon zeta"})
    run = Run("fake/sys", "calls-bench")
    store.hypotheses.write(run, {"calls/call-a": "alpha beta gamma", "calls/call-b": "epsilon"})
    assert store.test_cases.scored_per_session("calls-bench")
    rows = score(store, run, normalizers(paths).get("verbatim"))
    assert [(r["test_case_id"], r["n_ref"], r["n_del"]) for r in rows] == [("calls/call-a", 4, 1), ("calls/call-b", 2, 1)]
    assert score_summary.counts(store, run, rows) == {"test_cases": 2, "scored": 2, "empty": 0}


# ── the scores ───────────────────────────────────────────────────────────────────────────


def test_a_metric_is_empty_until_it_is_computed_and_is_then_saved_beside_the_others(paths):
    store = make_run(paths)
    normalizer = normalizers(paths).get("verbatim")
    words = score(store, TOY_RUN, normalizer, "wer")
    assert store.scores.has(TOY_RUN, "verbatim", "wer")
    assert not store.scores.has(TOY_RUN, "verbatim", "cer") and not store.scores.has(TOY_RUN, "verbatim", "semantic")
    chars = score(store, TOY_RUN, normalizer, "cer")
    assert (words[0]["n_ref"], words[0]["n_sub"]) == (3, 1) and chars[0]["n_ref"] == len("alphabetagamma")
    # The character errors are kept beside the word errors, which are as they were.
    assert store.scores.read(TOY_RUN, "verbatim", "cer") == chars and store.scores.read(TOY_RUN, "verbatim", "wer") == words
    with pytest.raises(KeyError, match="wer"):
        score(store, TOY_RUN, normalizer, "bleu")


def test_a_saved_metric_is_read_and_counted_again_only_when_asked(paths):
    store = make_run(paths)
    normalizer = normalizers(paths).get("verbatim")
    kept = score(store, TOY_RUN, normalizer)
    assert kept[0]["n_sub"] == 1
    kept[0]["n_sub"] = 99
    store.scores.write(TOY_RUN, "verbatim", "wer", kept)
    assert score(store, TOY_RUN, normalizer)[0]["n_sub"] == 99  # read, not counted
    assert score(store, TOY_RUN, normalizer, rescore=True)[0]["n_sub"] == 1
    assert store.scores.read(TOY_RUN, "verbatim", "wer")[0]["n_sub"] == 1
