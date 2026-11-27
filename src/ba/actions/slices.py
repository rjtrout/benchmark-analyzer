"""Slices: a metric broken down by a metadata field.

The slices are premade: one for each metadata field, with a number cut into bins. There is
no minimum count. A score is shown however few test cases are behind it, with how many test
cases and reference units it has beside it, so the reader can see what it rests on. The
test cases behind any one bin can be read too, as examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ba.actions.normalization import normalized_hypotheses, normalized_references
from ba.actions.scoring import score
from ba.aggregations.aggregations import error_rate
from ba.aggregations.registry import AGGREGATIONS
from ba.metadata.derived import FLAGS
from ba.normalize.normalizer import Normalizer
from ba.store.store import Store
from ba.store.transcriptions import Run

# The premade slices: the field each breaks a score down by, and what a chart calls it.
SLICES = {
    "duration": "Duration (s)",
    "sampling_rate": "Sample rate (Hz)",
    "level": "Level (dBFS)",
    "snr_given": "SNR, given (dB)",
    "snr_est": "SNR, derived (dB)",
    "bandwidth_est": "Bandwidth (Hz)",
    "speech_fraction": "Speech fraction",
    "ref_words": "Reference words",
    "speaking_rate": "Speaking rate (words/s)",
    "alphanumeric": "Alphanumeric present",
}
BINS = 10  # how many bins a number is cut into, unless told otherwise
EXAMPLES = 10  # how many examples of a bin are given at a time, unless told otherwise
NO_VALUE = "(no value)"


@dataclass
class Group:
    """The test cases of one bin of a number, or of one value of a flag."""

    label: str
    rows: list[dict[str, Any]]  # each test case's score row
    at: float | None = None  # where a bin is drawn: the mean of its test cases' values


def _bins(values: list[float], bins: int) -> list[tuple[str, float]]:
    """A number's bins, in rising order: each one's label and its lower edge.

    Each bin holds the same number of test cases. A bin holds its lower edge, and the last one
    its upper edge too. A number with no more values than bins, such as a sampling rate,
    gets a bin for each value instead: none is empty, and none is made up between two values.
    """
    distinct = sorted(set(values))
    if len(distinct) <= bins:
        return [(f"{value:g}", value) for value in distinct]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    found = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        found.append((f"{low:.2f} to {high:.2f}", float(low)))
    return found


def _bin_label(value: float, bins: list[tuple[str, float]]) -> str:
    """The bin a value is in: the last one whose lower edge it reaches."""
    lower_edges = [low for _, low in bins]
    index = int(np.searchsorted(lower_edges, value, side="right")) - 1
    return bins[index][0]


def _groups(
    store: Store,
    run: Run,
    normalizer: Normalizer,
    metric_id: str,
    by: str,
    bins: int,
) -> list[Group]:
    """A run's scored test cases, put into the bins of a number or the values of a flag.

    The groups come in rising order: a number's bins from the lowest, a flag's false and
    then true. Test cases with no value for the field are the last group. A group that
    holds no test case is left out.
    """
    kept = score(store, run, normalizer, metric_id)
    values = {}  # each test case's value for the field
    for test_case in store.test_cases.read(run.benchmark_id):
        values[test_case.test_case_id] = getattr(test_case, by)
    is_number = by not in FLAGS

    if is_number:
        known_values = [values[row["test_case_id"]] for row in kept if values[row["test_case_id"]] is not None]
        number_bins = _bins(known_values, bins)
        order = [label for label, _ in number_bins]
    else:
        order = ["false", "true"]

    groups = {}
    values_of: dict[str, list[float]] = {}
    for label in [*order, NO_VALUE]:
        groups[label] = Group(label, rows=[])
        values_of[label] = []
    for row in kept:
        value = values[row["test_case_id"]]
        if value is None:
            label = NO_VALUE
        elif is_number:
            label = _bin_label(value, number_bins)
        else:
            label = str(value).lower()  # "true" or "false"
        groups[label].rows.append(row)
        if value is not None:
            values_of[label].append(value)

    found = []
    for label, group in groups.items():
        if not group.rows:
            continue
        # A flag, and the test cases with no value, have no place on a number line.
        if is_number and label != NO_VALUE:
            group.at = float(np.mean(values_of[label]))
        found.append(group)
    return found


def slice_run(
    store: Store,
    run: Run,
    normalizer: Normalizer,
    metric_id: str = "wer",
    by: str = "duration",
    bins: int = BINS,
    agg: str = "overall",
) -> list[dict[str, Any]]:
    """One row per bin of a number, or per value of a flag: the score and what it rests on.

    Bins come in rising order, so the higher values are last, and ``at`` is where a bin is
    drawn: the mean of its test cases' values. A flag gives false, then true. Test cases with
    no value for the field come last, counted together. A metric that is not scored with the
    normalizer, such as semantic error, has to be stored already: this reads it, and does not
    compute it. The run is of a benchmark scored per test case: a session has no one value.
    """
    # An unknown field or aggregation is an error before any score is read.
    if by not in SLICES:
        raise KeyError(f"No field '{by}' to slice by. Choose one of: {', '.join(SLICES)}")
    aggregate = AGGREGATIONS.get(agg)

    sliced = []
    for group in _groups(store, run, normalizer, metric_id, by, bins):
        sliced.append(
            {
                "value": group.label,
                "at": group.at,
                "test_cases": len(group.rows),
                "ref_units": sum(row["n_ref"] for row in group.rows),
                "score": aggregate(group.rows),
            }
        )
    return sliced


def examples(
    store: Store,
    run: Run,
    normalizer: Normalizer,
    metric_id: str = "wer",
    by: str = "duration",
    bins: int = BINS,
    value: str = NO_VALUE,
    highest_first: bool = True,
    start: int = 0,
    count: int = EXAMPLES,
) -> tuple[list[dict[str, Any]], int]:
    """Test cases of one bin or value of a slice, each with its score and the two texts it compares.

    ``value`` names the bin or value, as a row of ``slice_run`` does. Its test cases are put
    in the order of their own score, the highest first unless told otherwise, and ``count``
    of them are given, from place ``start``. Returns them, and how many test cases the bin
    holds in all. The texts are as the normalizer leaves them: what the score is of.
    """
    rows = []
    for group in _groups(store, run, normalizer, metric_id, by, bins):
        if group.label == value:
            rows = group.rows
    in_order = sorted(rows, key=error_rate, reverse=highest_first)  # test cases level on score stay in list order
    shown = in_order[start : start + count]

    references = normalized_references(store, run.benchmark_id, normalizer)
    hypotheses = normalized_hypotheses(store, run, normalizer)
    found = []
    for row in shown:
        test_case_id = row["test_case_id"]
        found.append(
            {
                "test_case_id": test_case_id,
                "score": 100.0 * error_rate(row),
                "reference": references[test_case_id],
                "hypothesis": hypotheses[test_case_id],
            }
        )
    return found, len(rows)
