"""Results: the error rate of each model, with a line per metric.

The models are on the x-axis and the lines are WER, CER and semantic error. Pick the
normalizer the scores are under, how they are aggregated (mean, median or overall), the
benchmarks and the models. With several benchmarks, a large chart of the combined score
comes first, and a chart of each benchmark follows. Semantic error can be computed from here.
"""

from __future__ import annotations

import streamlit as st

from ba.ui import compare, controls, page, queries


def main() -> None:
    project = page.frame()
    st.title("Results")
    st.caption("The error rate of each model, with a line per metric. Every model on a benchmark is scored on the same test cases.")

    picked = compare.chosen(project, "results", first="normalizer")
    if picked is None:
        return
    choices, runs = picked

    compare.semantic_panel(project, runs, "results")
    rows = queries.metric_rows(project, runs, choices.normalizer_id, choices.agg)
    compare.draw(
        project,
        rows,
        series="metric",
        series_title="Metric",
        series_order=list(controls.METRICS),
        choices=choices,
        y_title=f"Error rate, {choices.agg} (%)",
    )


with page.guard():
    main()
