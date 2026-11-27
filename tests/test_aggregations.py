"""The aggregations: mean, median and overall, each from the rows of test cases."""

from __future__ import annotations

import pytest

from ba.aggregations.aggregations import error_rate, summarize
from ba.aggregations.registry import AGGREGATIONS
from tests.conftest import row


def test_aggregations_are_all_computed_from_per_test_case_counts():
    rows = [row("a", 10, n_sub=1), row("b", 10), row("c", 2, n_del=2), row("d", 10), row("e", 10)]
    out = summarize(rows, AGGREGATIONS.ids())
    assert out["overall"] == pytest.approx(100 * 3 / 42)
    assert out["mean"] == pytest.approx(100 * (0.1 + 1.0) / 5)
    assert out["median"] == 0.0


def test_a_metric_stored_as_a_rate_has_the_same_three_aggregations():
    # Semantic error: a rate per test case, with the reference's words as its weight.
    rows = [row("a", 10, rate=0.1), row("b", 30, rate=0.5), row("c", 2, rate=0.0)]
    out = summarize(rows, ["mean", "median", "overall"])
    assert out["mean"] == pytest.approx(100 * (0.1 + 0.5 + 0.0) / 3)
    assert out["median"] == pytest.approx(10.0)
    # Overall: every test case's rate weighted by its words, as overall WER weights a test case by its words.
    assert out["overall"] == pytest.approx(100 * (0.1 * 10 + 0.5 * 30 + 0.0 * 2) / 42)


def test_a_test_case_with_no_reference_words_is_all_wrong_if_anything_was_written():
    assert error_rate(row("a", 4, n_sub=1)) == pytest.approx(0.25)
    assert error_rate(row("b", 2, n_ins=3)) == pytest.approx(1.5)  # words added can take it past 1.0
    assert error_rate(row("c", 0, n_ins=2)) == 1.0 and error_rate(row("d", 0)) == 0.0
    assert error_rate(row("e", 10, rate=0.3)) == pytest.approx(0.3)  # a metric stored as a rate
