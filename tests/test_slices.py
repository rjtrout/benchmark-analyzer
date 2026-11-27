"""Slices: a metric broken down by a metadata field, with counts beside every score."""

from __future__ import annotations

import pytest

from ba.actions import scoring, slices
from ba.metadata.derived import DERIVED
from ba.metadata.given import GIVEN
from ba.metadata.populate import populate_metadata
from ba.normalize.registry import normalizers
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run

DATA = [  # source id, duration, reference, hypothesis
    ("AMI_EN2001a_H03_MEE067_0000100_0000200", 1.0, "alpha beta gamma delta", "alpha beta gamma delta"),
    ("AMI_EN2001a_H03_MEE067_0000300_0000400", 2.0, "alpha beta gamma delta", "alpha beta gamma"),
    ("AMI_EN2001a_H01_FEO065_0000500_0000600", 6.0, "is it twenty five", "is it 25"),
    ("AMI_ES2004a_H02_MED012_0000700_0000800", 9.0, "alpha beta gamma delta", "wrong wrong wrong wrong"),
]


R1 = Run("fake/sys", "ami-cleaned")


@pytest.fixture
def store(paths) -> Store:
    store = Store(paths)
    test_cases = []
    references = {}
    for sid, duration, ref, _ in DATA:
        test_case = TestCase(f"ami_cleaned/{sid}")
        populate_metadata(test_case, {"audio_length_s": duration}, ref, paths)
        test_cases.append(test_case)
        references[test_case.test_case_id] = ref
    store.test_cases.write("ami-cleaned", test_cases)
    store.references.write("ami-cleaned", references)
    store.hypotheses.write(R1, {f"ami_cleaned/{sid}": hyp for sid, _, _, hyp in DATA})
    return store


def cut(store, paths, **kwargs):
    return slices.slice_run(store, R1, normalizers(paths).get("leaderboard"), **kwargs)


def test_every_metadata_field_has_a_premade_slice():
    assert set(slices.SLICES) == set(GIVEN) | set(DERIVED)


def test_a_number_is_cut_into_bins_that_hold_every_test_case(store, paths):
    rows = cut(store, paths, by="duration", bins=2)
    assert [r["test_cases"] for r in rows] == [2, 2]
    assert sum(r["ref_units"] for r in rows) == 15
    assert rows[0]["value"].startswith("1.00 to") and rows[1]["value"].endswith("9.00")
    # A bin is drawn where its test cases are: at the mean of their values.
    assert [r["at"] for r in rows] == [1.5, 7.5]
    assert set(rows[0]) == {"value", "at", "test_cases", "ref_units", "score"}  # the score, where it is drawn, and what it rests on
    assert rows[0]["score"] == pytest.approx(100 * 1 / 8)  # the two short test cases: one word of eight is missing


def test_a_number_with_fewer_values_than_bins_gets_a_bin_for_each_value(store, paths):
    # Four test cases of four lengths, and ten bins asked for: no bin is empty, and none is made up between two values.
    rows = cut(store, paths, by="duration", bins=10)
    assert [(r["value"], r["at"], r["test_cases"]) for r in rows] == [("1", 1.0, 1), ("2", 2.0, 1), ("6", 6.0, 1), ("9", 9.0, 1)]


def test_a_flag_splits_in_two(store, paths):
    rows = {r["value"]: r for r in cut(store, paths, by="alphanumeric")}
    assert rows["true"]["test_cases"] == 1 and rows["false"]["test_cases"] == 3
    assert rows["true"]["score"] == 0.0  # "twenty five" and "25" are the same number
    assert rows["true"]["at"] is None  # a flag has no place on a number line


def test_test_cases_with_no_value_are_counted_together_and_never_dropped(store, paths):
    # AMI states no noise level. The score is still shown, with what it rests on.
    assert cut(store, paths, by="snr_given") == [{"value": "(no value)", "at": None, "test_cases": 4, "ref_units": 15, "score": pytest.approx(100 * 5 / 15)}]
    # A field measured on the audio is the same before the audio is fetched: no test case has a value yet.
    assert cut(store, paths, by="level") == cut(store, paths, by="snr_given")


def test_an_unknown_field_names_the_real_ones(store, paths):
    with pytest.raises(KeyError, match="duration"):
        cut(store, paths, by="loudness")


def test_bins_and_flags_come_in_rising_order(store, paths):
    # A chart puts them on its x-axis in this order: the higher values are further right.
    rows = cut(store, paths, by="duration", bins=3)
    durations = [row["value"] for row in rows]
    assert durations == sorted(durations, key=lambda label: float(label.split(" to ")[0])) and len(durations) == 3
    # Lengths of 1, 2, 6 and 9 seconds in bins that meet at 2 and at 6: a test case on the line between two bins
    # is in the higher one, and the last bin holds its upper edge too.
    assert [row["test_cases"] for row in rows] == [1, 1, 2]
    assert [row["value"] for row in cut(store, paths, by="alphanumeric")] == ["false", "true"]


def with_semantic(store, normalizer, rates) -> None:
    """Put a semantic error on each of the run's rows, as ``ba semantic`` would."""
    rows = scoring.score(store, R1, normalizer, "wer")
    for row, rate in zip(rows, rates, strict=True):
        row["rate"] = rate
    store.scores.write(R1, normalizer.id, "semantic", rows)


def test_a_metric_computed_on_request_is_read_and_never_computed_by_a_slice(store, paths):
    normalizer = normalizers(paths).get("leaderboard")
    with pytest.raises(scoring.NotComputedError, match="ba semantic"):
        slices.slice_run(store, R1, normalizer, "semantic", by="duration")
    # Once it is stored, it is sliced like any other metric: here a rate of 0.5 on every test case.
    with_semantic(store, normalizer, [0.5, 0.5, 0.5, 0.5])
    sliced = slices.slice_run(store, R1, normalizer, "semantic", by="duration", bins=2, agg="mean")
    assert [row["score"] for row in sliced] == [pytest.approx(50.0), pytest.approx(50.0)]


# ── examples: the test cases behind one bin or value ─────────────────────────────────────


def examples(store, paths, normalizer_id="leaderboard", **kwargs):
    return slices.examples(store, R1, normalizers(paths).get(normalizer_id), **kwargs)


def test_examples_are_the_test_cases_of_one_bin_with_the_worst_first(store, paths):
    low, high = [row["value"] for row in cut(store, paths, by="duration", bins=2)]
    rows, in_all = examples(store, paths, by="duration", bins=2, value=low, count=2)
    # The two short test cases. One word of four is missing from the second, so it comes first.
    assert in_all == 2
    assert [(row["score"], row["reference"], row["hypothesis"]) for row in rows] == [
        (25.0, "alpha beta gamma delta", "alpha beta gamma"),
        (0.0, "alpha beta gamma delta", "alpha beta gamma delta"),
    ]
    assert [row["test_case_id"] for row in rows] == [f"ami_cleaned/{DATA[1][0]}", f"ami_cleaned/{DATA[0][0]}"]
    rows, in_all = examples(store, paths, by="duration", bins=2, value=high, count=2)
    assert [row["score"] for row in rows] == [100.0, 0.0] and in_all == 2


def test_examples_can_be_put_best_first_and_taken_a_few_at_a_time(store, paths):
    # Every reference here is four words long, so a slice by reference words has one bin, of all four.
    every, in_all = examples(store, paths, by="ref_words", value="4", highest_first=False, count=4)
    assert in_all == 4 and [row["score"] for row in every] == [0.0, 0.0, 25.0, 100.0]
    # Two at a time: the second two, and nothing past the end.
    second, in_all = examples(store, paths, by="ref_words", value="4", highest_first=False, start=2, count=2)
    assert [row["score"] for row in second] == [25.0, 100.0] and in_all == 4
    assert examples(store, paths, by="ref_words", value="4", start=4, count=2) == ([], 4)


def test_examples_show_the_texts_as_the_normalizer_leaves_them(store, paths):
    # Under the leaderboard's rules "twenty five" is 25, and nothing differs.
    [row], _ = examples(store, paths, by="alphanumeric", value="true")
    assert (row["reference"], row["hypothesis"], row["score"]) == ("is it 25", "is it 25", 0.0)
    # As written, two of the four reference words are not in the transcript.
    [row], _ = examples(store, paths, normalizer_id="verbatim", by="alphanumeric", value="true")
    assert (row["reference"], row["hypothesis"], row["score"]) == ("is it twenty five", "is it 25", 50.0)


def test_examples_of_a_bin_that_is_not_there_are_none(store, paths):
    assert examples(store, paths, by="duration", bins=2, value="no such bin") == ([], 0)


def test_examples_follow_a_metric_that_is_stored_as_a_rate(store, paths):
    with_semantic(store, normalizers(paths).get("leaderboard"), [0.1, 0.4, 0.2, 0.3])
    found, _ = examples(store, paths, metric_id="semantic", by="ref_words", value="4", count=4)
    assert [row["score"] for row in found] == pytest.approx([40.0, 30.0, 20.0, 10.0])
