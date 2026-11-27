"""What the Results and Normalization pages share: the error rate against the models.

The models are on the x-axis and each line is a series: a metric on Results, a normalizer on
Normalization. With several benchmarks chosen, a large chart of the combined score comes
first, and a smaller chart of each benchmark follows.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from ba.metrics import semantic as semantic_metric
from ba.store.transcriptions import Run
from ba.ui import charts, controls, page, queries

NOTE = "semantic_note"  # where a note is kept for the page's next run, after semantic error is computed


def chosen(project: page.Project, key: str, first: str) -> tuple[controls.Choices, list[Run]] | None:
    """The page's controls and the runs they pick, or None after saying what is missing."""
    if not controls.benchmarks_with_runs(project):
        st.info("Nothing is fetched yet. `ba fetch` reads each benchmark's test case list and each model's public transcripts.")
        return None
    choices = controls.choose(project, key, first=first)
    if not choices.benchmark_ids:
        st.info("Pick at least one benchmark.")
        return None
    if not choices.system_ids:
        st.info("Pick at least one model.")
        return None
    return choices, controls.chosen_runs(project, choices)


def semantic_panel(project: page.Project, runs: list[Run], key: str) -> None:
    """Say what has no semantic error yet, and offer to compute it."""
    note = st.session_state.pop(NOTE, None)
    if note:
        st.success(note)

    per_session = set()
    for run in runs:
        if project.store.test_cases.scored_per_session(run.benchmark_id):
            per_session.add(run.benchmark_id)
    if per_session:
        st.caption(
            f"No semantic error for {', '.join(sorted(per_session))}: it is scored per session, and a whole session is far more text than a sentence encoder reads."
        )

    missing = queries.semantic_missing(project, runs)
    if not missing:
        return
    st.info(
        f"Semantic error is not computed yet for {len(missing)} model and normalizer pairs on these benchmarks. It is the distance between the "
        f"sentence embeddings of the reference and the transcript, from `{semantic_metric.MODEL}`, and it runs on this machine."
    )
    if not st.button("Compute semantic error", key=f"{key}_semantic", type="primary"):
        return

    bar = st.progress(0.0)

    def show_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Semantic error: {done} of {total} model and normalizer pairs done.")

    try:
        done = queries.compute_semantic(project, missing, progress=show_progress)
    finally:
        bar.empty()
    st.session_state[NOTE] = f"Semantic error computed for {done} model and normalizer pairs."
    st.rerun()


def draw(
    project: page.Project,
    rows: queries.Rows,
    series: str,
    series_title: str,
    series_order: Sequence[str],
    choices: controls.Choices,
    y_title: str,
) -> None:
    """The charts and the table for these rows: the combined score first when there is more than one benchmark."""
    labels = controls.model_labels(project)

    def draw_chart(data: queries.Rows, large: bool) -> None:
        chart = charts.model_lines(
            data,
            series,
            series_title,
            series_order,
            choices.system_ids,
            labels,
            y_title,
            large=large,
        )
        st.altair_chart(chart, width="content")

    combined_rows: queries.Rows = []
    try:
        if len(choices.benchmark_ids) > 1:
            title = f"Combined score: {len(choices.benchmark_ids)} benchmarks"
            combined_rows, notes = queries.combined(rows, series, title)
            draw_chart(combined_rows, large=True)
            st.caption("The combined score is the mean of the chosen benchmarks' figures. Each benchmark counts the same, as in the leaderboard's own average.")
            for note in notes:
                st.caption(note)
            draw_chart(rows, large=False)
        else:
            draw_chart(rows, large=True)
    except charts.NoData as exc:
        st.info(str(exc))
        return
    table(rows, combined_rows, series, series_order)


def table(rows: queries.Rows, combined_rows: queries.Rows, series: str, series_order: Sequence[str]) -> None:
    """One row per model and benchmark, a column per series. The combined score comes first."""
    # Each row of the table, by its model and benchmark. A row is started the first time one of its figures is met.
    wide: dict[tuple[str, str], dict] = {}
    for row in combined_rows:
        started = {"model": row["system"], "benchmark": "combined"}
        entry = wide.setdefault((row["system"], "combined"), started)
        entry[row[series]] = row["value"]
        entry["scored / test cases"] = ""
    for row in rows:
        started = {"model": row["system"], "benchmark": row["benchmark"]}
        entry = wide.setdefault((row["system"], row["benchmark"]), started)
        entry[row[series]] = row["value"]
        entry["scored / test cases"] = f"{row['scored']}/{row['test_cases']}"

    # A column for each series that has a figure somewhere, in the series' own order.
    present = []
    for name in series_order:
        if any(name in entry for entry in wide.values()):
            present.append(name)
    columns = ["model", "benchmark", *present, "scored / test cases"]

    ordered = []
    for entry in wide.values():
        ordered.append({column: entry.get(column) for column in columns})
    page.table(ordered, decimals=present)
