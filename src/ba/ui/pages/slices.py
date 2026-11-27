"""Slices: the error rate broken down by something known about each test case.

Two premade slices side by side, each of a benchmark of its own, with a line per model. A
number is cut into bins, drawn where their test cases are, so the higher values are further right.
Under each chart are the numbers behind it, and examples: the test cases of any one bin, ten
at a time, each with its reference and what a model wrote for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import streamlit as st

from ba.actions import slices
from ba.metadata.derived import FLAGS, FROM_AUDIO
from ba.ui import charts, controls, page, queries
from ba.utils.list_utils import unique

# A metric's id -> what the controls call it.
METRIC_NAMES = {metric_id: label for label, metric_id in controls.METRICS.items()}
# Each side, and the slice it opens on: two that need no audio.
SIDES = {"left": "duration", "right": "ref_words"}
# What the order of the examples is called -> whether the highest score comes first.
ORDERS = {"Highest first": True, "Lowest first": False}


@dataclass
class Shared:
    """What the two sides share: the choices made at the top of the page."""

    metric_id: str
    agg: str
    normalizer_id: str
    bins: int
    system_ids: list[str]


def examples(
    project: page.Project,
    key: str,
    by: str,
    benchmark_id: str,
    rows: queries.Rows,
    shared: Shared,
) -> None:
    """One model's test cases of one bin or value of the slice, ten at a time.

    Each has its own score, its reference and the model's hypothesis. ``rows`` are the
    slice's rows: they say which bins there are, and how many test cases each holds.
    """
    labels = controls.model_labels(project)

    def label(system_id: str) -> str:
        return labels[system_id]

    # Which model, and which bin or value of the slice.
    first, second = st.columns(2)
    system_ids = unique(row["system"] for row in rows)
    system_id = first.selectbox("Model", system_ids, format_func=label, key=f"slices_{key}_examples_model")
    of_the_model = [row for row in rows if row["system"] == system_id]
    value = second.selectbox(
        slices.SLICES[by],
        [row["value"] for row in of_the_model],
        key=f"slices_{key}_examples_value",
        help="The bin or value of the slice the examples are taken from.",
    )
    in_all = 0
    for row in of_the_model:
        if row["value"] == value:
            in_all = row["test_cases"]

    # In which order, and which ten.
    third, fourth = st.columns(2)
    order = third.radio("Order", list(ORDERS), horizontal=True, key=f"slices_{key}_examples_order")
    pages = max(1, math.ceil(in_all / slices.EXAMPLES))
    # A page number belongs to what is shown: another benchmark, bin, model or order starts at its first page.
    page_number = fourth.number_input(
        "Page",
        min_value=1,
        max_value=pages,
        value=1,
        step=1,
        key=f"slices_{key}_examples_page_{benchmark_id}_{by}_{system_id}_{value}_{order}",
        help=f"{slices.EXAMPLES} test cases to a page.",
    )
    start = (int(page_number) - 1) * slices.EXAMPLES

    shown = queries.slice_examples(
        project.root,
        project.stamp,
        system_id,
        benchmark_id,
        shared.normalizer_id,
        shared.metric_id,
        by,
        shared.bins,
        value,
        ORDERS[order],
        start,
    )

    metric = METRIC_NAMES[shared.metric_id]
    st.caption(
        f"{start + 1} to {start + len(shown)} of {in_all:,} test cases, by each one's own {metric}. "
        f"The texts are shown as the {shared.normalizer_id} normalizer leaves them, which is what the score compares."
    )
    table = []
    for row in shown:
        table.append({f"{metric} (%)": f"{row['score']:.2f}", "reference": row["reference"], "hypothesis": row["hypothesis"]})
    page.wrapped_table(table)


def side(project: page.Project, key: str, opens_on: str, offered: list[str], shared: Shared) -> None:
    """One slice of one benchmark: its two choices, its chart, the numbers behind the chart, and examples."""
    premade = list(slices.SLICES)
    first, second = st.columns(2)
    by = first.selectbox(
        "Slice",
        premade,
        index=premade.index(opens_on),
        format_func=slices.SLICES.get,
        key=f"slices_{key}_by",
    )
    benchmark_id = second.selectbox("Benchmark", offered, key=f"slices_{key}_benchmark")
    title = slices.SLICES[by]

    runs = [run for run in controls.runs(project, [benchmark_id]) if run.system_id in shared.system_ids]
    rows, missing = queries.slice_rows(
        project,
        runs,
        shared.normalizer_id,
        shared.metric_id,
        by,
        shared.bins,
        shared.agg,
    )
    if missing:
        st.info(f"Semantic error is not computed yet for {len(missing)} of these models. Compute it on the Results page, or with `ba semantic`.")
    if not rows:
        return

    y_title = f"{METRIC_NAMES[shared.metric_id]}, {shared.agg} (%)"
    # Test cases with no value for the field have no place on the chart. They are in the numbers below it.
    drawn = [row for row in rows if row["value"] != slices.NO_VALUE]
    try:
        chart = charts.slice_lines(drawn, y_title, title, controls.model_colours(project), number=by not in FLAGS)
    except charts.NoData:
        note = f"No test case of {benchmark_id} has a value for {title}."
        if by in FROM_AUDIO:
            note += f" It is measured on the audio: `ba fetch --audio --benchmark {benchmark_id}` fetches it."
        st.info(note)
        return
    st.altair_chart(chart, width="stretch")

    numbers = []
    for row in rows:
        numbers.append(
            {
                "model": row["system"],
                title: row["value"],
                y_title: row["score"],
                "test cases": row["test_cases"],
                "reference units": row["reference units"],
            }
        )
    with st.expander("The numbers"):
        page.table(numbers, decimals=[y_title])
    with st.expander("Examples"):
        examples(project, key, by, benchmark_id, rows, shared)


def main() -> None:
    project = page.frame()
    st.title("Slices")
    st.caption("The error rate broken down by a metadata field, with a line per model. Two slices side by side, each of a benchmark of its own. Read the test case counts with the scores.")

    offered, per_session = controls.per_test_case_benchmarks(project)
    if not offered:
        st.info("Nothing to break down yet. `ba fetch` reads each benchmark's test case list and each model's public transcripts.")
        return

    # What the two sides share.
    one, two, three, four = st.columns([2, 2, 2, 1])
    metric_id = controls.pick_metric(one, "slices")
    agg = controls.pick_aggregation(two, "slices")
    normalizer_id = controls.pick_normalizer(three, project, "slices")
    bins = four.number_input(
        "Bins",
        min_value=2,
        max_value=200,
        value=slices.BINS,
        step=1,
        key="slices_bins",
        help="For a number: how many bins, each holding the same number of test cases.",
    )
    system_ids = controls.pick_models(project, "slices")
    if per_session:
        st.caption(f"Left out: {', '.join(per_session)}. It is scored per session, so there is no test case to break down by.")
    if not system_ids:
        st.info("Pick at least one model.")
        return
    shared = Shared(metric_id=metric_id, agg=agg, normalizer_id=normalizer_id, bins=int(bins), system_ids=system_ids)

    # One legend for the two charts.
    colours = controls.model_colours(project)
    labels = controls.model_labels(project)
    page.legend({labels[model]: colours[model] for model in system_ids})

    left, right = st.columns(2)
    columns = {"left": left, "right": right}
    for key, opens_on in SIDES.items():
        with columns[key]:
            side(project, key, opens_on, offered, shared)


with page.guard():
    main()
