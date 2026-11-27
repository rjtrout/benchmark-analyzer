"""Aggregations: one figure for a set of test cases, in percent, from the row of each.

A row of WER or CER holds counts, and its error rate is errors over reference units. A row
of semantic error holds the rate itself, in ``rate``, with the reference's word count as its
weight. So the same three aggregations read every metric:

mean     The mean of the test cases' error rates. Every test case counts the same.
median   The middle test case's error rate.
overall  All errors over all reference units. For semantic error, the mean of the test cases'
         rates weighted by their reference words.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Rows = list[dict[str, Any]]


def errors(row: dict[str, Any]) -> float:
    """A test case's errors, in reference units. For a metric stored as a rate: the rate times its weight."""
    if row.get("rate") is not None:
        return row["rate"] * row["n_ref"]
    return row["n_sub"] + row["n_del"] + row["n_ins"]


def error_rate(row: dict[str, Any]) -> float:
    """One test case's error rate, as a fraction. It can pass 1.0 when words are inserted.

    A test case with no reference words is 1.0 if the system wrote any words and 0.0 if not.
    """
    if row.get("rate") is not None:
        return float(row["rate"])
    if row["n_ref"] == 0:
        if row["n_ins"] > 0:
            return 1.0
        return 0.0
    return errors(row) / row["n_ref"]


def _rates(rows: Rows) -> np.ndarray:
    return np.array([error_rate(row) for row in rows], dtype=float)


def overall(rows: Rows) -> float:
    """All the test cases' errors over all their reference units, in percent."""
    reference_units = sum(row["n_ref"] for row in rows)
    total_errors = sum(errors(row) for row in rows)
    if reference_units == 0:
        # Nothing was said. Any word written is then an error with nothing to measure it against.
        if total_errors == 0:
            return 0.0
        return float("inf")
    return 100.0 * total_errors / reference_units


def mean(rows: Rows) -> float:
    """The mean of the test cases' error rates, in percent. Not a number when there are no test cases."""
    if not rows:
        return float("nan")
    return 100.0 * float(_rates(rows).mean())


def median(rows: Rows) -> float:
    """The middle test case's error rate, in percent. Not a number when there are no test cases."""
    if not rows:
        return float("nan")
    return 100.0 * float(np.median(_rates(rows)))


# The aggregations by id, in the order every table and every page shows them. ``registry.py`` makes its table of these.
BY_ID = {"mean": mean, "median": median, "overall": overall}


def summarize(rows: Rows, aggregation_ids: list[str]) -> dict[str, float]:
    """The asked-for aggregations of these rows, by id."""
    figures = {}
    for aggregation_id in aggregation_ids:
        aggregate = BY_ID[aggregation_id]
        figures[aggregation_id] = aggregate(rows)
    return figures
