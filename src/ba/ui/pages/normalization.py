"""Normalization: the error rate of each model, with a line per normalizer.

The same chart as Results, with the normalizers as the lines: how far apart they are is what
the rules forgive. Pick the metric, how it is aggregated, the benchmarks and the models.
Below, two heatmaps show what each rule family is worth, so the rules that help the most
stand out: first the leaderboard's rules, ``leaderboard_split``, and under it our own,
``own``, which has four more. A family switched off is a normalizer like any other: its
scores are saved the first time they are wanted, and read after that.
"""

from __future__ import annotations

import streamlit as st

from ba.normalize.registry import FAMILIES_OF
from ba.store.transcriptions import Run
from ba.ui import charts, compare, controls, page, queries

# A metric's id -> what the controls call it.
METRIC_NAMES = {metric_id: label for label, metric_id in controls.METRICS.items()}

# The normalizers that are made of rule families, in the order their heatmaps are drawn, and what each is.
MADE_OF_FAMILIES = {
    "leaderboard_split": "the leaderboard's rules, as families that switch off",
    "own": "those rules and four more of ours: cut-off words, discourse markers, interjections and repetitions",
}


def family_heatmap(
    project: page.Project,
    runs: list[Run],
    normalizer_id: str,
    metric_id: str,
    agg: str,
    combine_benchmarks: bool,
) -> None:
    """The heatmap of what each rule family of one normalizer is worth."""
    # A model and benchmark not scored yet under each family is scored now, once for each family, and saved.
    with st.spinner(f"Working out what each rule family of {normalizer_id} helps ..."):
        rows = queries.ablation_rows(project, runs, normalizer_id, metric_id, agg)
    if metric_id == "semantic" and not rows:
        st.info(f"Semantic error is not computed for the rule families of `{normalizer_id}` yet. Compute it with: `ba semantic --families`")
        return
    if combine_benchmarks:
        rows, notes = queries.combined(rows, "family", "Combined", value_fields=("change", "with every family on"))
        for note in notes:
            st.caption(note)

    order = list(FAMILIES_OF[normalizer_id])
    try:
        # Its own width, not the page's: a cell keeps its size however many families there are.
        heatmap = charts.family_heatmap(rows, order, dark=page.is_dark(), combined=combine_benchmarks)
        st.altair_chart(heatmap, width="content")
    except charts.NoData as exc:
        st.info(str(exc))


def families(project: page.Project, runs: list[Run], metric_id: str, agg: str) -> None:
    """What each rule family is worth: a heatmap for the leaderboard's rules, and under it one for our own."""
    st.subheader("What each rule family helps")
    st.caption(
        "Each cell is the rise in the error rate, in points, when one family is off and the rest stay on. "
        "The larger the number, the more that family helps the score."
    )
    combine_benchmarks = st.toggle(
        "Combine benchmarks",
        key="normalization_combine_benchmarks",
        help="Average the selected benchmarks shared by all models in the heatmaps, giving each benchmark equal weight.",
    )
    for normalizer_id, what in MADE_OF_FAMILIES.items():
        st.markdown(f"**`{normalizer_id}`**: {what}.")
        family_heatmap(project, runs, normalizer_id, metric_id, agg, combine_benchmarks)


def main() -> None:
    project = page.frame()
    st.title("Normalization")
    st.caption("The error rate of each model, with a line per normalizer. The gap between two lines is what the rules between them forgive.")

    picked = compare.chosen(project, "normalization", first="metric")
    if picked is None:
        return
    choices, runs = picked

    if choices.metric_id == "semantic":
        compare.semantic_panel(project, runs, "normalization")
    rows = queries.normalizer_rows(project, runs, choices.metric_id, choices.agg)
    compare.draw(
        project,
        rows,
        series="normalizer",
        series_title="Normalizer",
        series_order=controls.normalizer_ids(project),
        choices=choices,
        y_title=f"{METRIC_NAMES[choices.metric_id]}, {choices.agg} (%)",
    )
    families(project, runs, choices.metric_id, choices.agg)


with page.guard():
    main()
