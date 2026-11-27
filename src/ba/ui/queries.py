"""What the pages ask of the stored data: the rows behind each chart and table.

A read that is slow is cached, and keyed by the project's root and the stamp of its files.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import streamlit as st

from ba.actions import ablation, score_summary, scoring, slices
from ba.aggregations.aggregations import summarize
from ba.aggregations.registry import AGGREGATIONS
from ba.metrics import semantic as semantic_metric
from ba.normalize.registry import normalizers
from ba.store.transcriptions import Run
from ba.ui.controls import METRICS, normalizer_ids
from ba.ui.page import CACHE, Project, open_project
from ba.utils.list_utils import unique

Rows = list[dict[str, Any]]
Progress = Callable[[int, int], None]  # called with how many are done, and how many there are

# ── scores ───────────────────────────────────────────────────────────────────────────────


@st.cache_data(**CACHE)
def _run_scores(
    root: str,
    stamp: str,
    system_id: str,
    benchmark_id: str,
    metric_id: str,
    normalizer_ids: tuple[str, ...],
) -> Rows:
    """One row per normalizer for a run: the three aggregations and the counts beside them.

    A normalizer with no stored semantic error gets a row with ``missing`` set and no numbers.
    """
    paths, _, store = open_project(root)
    registry = normalizers(paths)
    run = Run(system_id, benchmark_id)
    out = []
    for normalizer_id in normalizer_ids:
        row = {"run_id": run.run_id, "system": system_id, "benchmark": benchmark_id, "normalizer": normalizer_id}
        try:
            scores = scoring.score(store, run, registry.get(normalizer_id), metric_id)
        except scoring.NotComputedError:
            row["missing"] = True
            out.append(row)
            continue
        row["missing"] = False
        row.update(score_summary.counts(store, run, scores))
        row["ref_units"] = sum(score["n_ref"] for score in scores)
        row.update(summarize(scores, AGGREGATIONS.ids()))
        out.append(row)
    return out


def result_rows(project: Project, runs: Sequence[Run], metric_id: str) -> Rows:
    """Every chosen run under every normalizer."""
    every_normalizer = tuple(normalizer_ids(project))
    rows: Rows = []
    for run in runs:
        rows.extend(_run_scores(project.root, project.stamp, run.system_id, run.benchmark_id, metric_id, every_normalizer))
    return rows


def _point(row: dict[str, Any], agg: str, series: str, name: str) -> dict[str, Any]:
    """A point of a chart: a run's figure, and the line it is on. ``series`` is what the lines are, and ``name`` is this one."""
    return {
        "system": row["system"],
        "benchmark": row["benchmark"],
        series: name,
        "value": row[agg],
        "scored": row["scored"],
        "test_cases": row["test_cases"],
    }


def metric_rows(project: Project, runs: Sequence[Run], normalizer_id: str, agg: str) -> Rows:
    """One row per run and metric, under one normalizer. A metric that is not computed for a run has no row."""
    out: Rows = []
    for label, metric_id in METRICS.items():
        for row in result_rows(project, runs, metric_id):
            if row["normalizer"] == normalizer_id and not row["missing"]:
                out.append(_point(row, agg, "metric", label))
    return out


def normalizer_rows(project: Project, runs: Sequence[Run], metric_id: str, agg: str) -> Rows:
    """One row per run and normalizer, for one metric."""
    out: Rows = []
    for row in result_rows(project, runs, metric_id):
        if not row["missing"]:
            out.append(_point(row, agg, "normalizer", row["normalizer"]))
    return out


def combined(
    rows: Rows, series: str, title: str, *, value_fields: tuple[str, ...] = ("value",)
) -> tuple[Rows, list[str]]:
    """Average the requested fields across benchmarks for each model and series.

    Each benchmark counts the same, as in the leaderboard's own average. For a series, only
    the benchmarks that every model in ``rows`` has a figure for go into it, so the models
    are compared on the same sets. Returns the rows, under ``title`` as their benchmark, and
    a note for each benchmark left out.
    """
    systems = unique(row["system"] for row in rows)
    benchmarks = unique(row["benchmark"] for row in rows)
    names = unique(row[series] for row in rows)
    figures = {}
    for row in rows:
        figures[(row["system"], row["benchmark"], row[series])] = row

    out: Rows = []
    left_out: dict[tuple[tuple[str, ...], str], list[str]] = {}  # (the benchmarks, why they are left out) -> the series
    for name in names:
        # Group the benchmarks by how many of the models have a figure there for this series.
        every_model = []
        some_models = []
        no_model = []
        for benchmark in benchmarks:
            have = [system for system in systems if (system, benchmark, name) in figures]
            if len(have) == len(systems):
                every_model.append(benchmark)
            elif have:
                some_models.append(benchmark)
            else:
                no_model.append(benchmark)

        if no_model:
            left_out.setdefault((tuple(no_model), "no model has a figure"), []).append(name)
        if some_models:
            left_out.setdefault((tuple(some_models), "not every chosen model has a figure"), []).append(name)
        if not every_model:
            continue
        for system in systems:
            matched = [figures[(system, benchmark, name)] for benchmark in every_model]
            averages = {field: sum(row[field] for row in matched) / len(matched) for field in value_fields}
            out.append(
                {
                    "system": system,
                    "benchmark": title,
                    series: name,
                    **averages,
                    "sets": len(every_model),
                }
            )

    notes = []
    for (missing, reason), series_names in left_out.items():
        verb = "is" if len(missing) == 1 else "are"
        notes.append(f"{', '.join(series_names)}: {', '.join(missing)} {verb} left out of the combined score, where {reason}.")
    return out, notes


# ── what each rule family is worth ───────────────────────────────────────────────────────


@st.cache_data(**CACHE)
def _run_ablation(root: str, stamp: str, system_id: str, benchmark_id: str, normalizer_id: str, metric_id: str, agg: str) -> Rows:
    _, _, store = open_project(root)
    return ablation.ablate(store, [Run(system_id, benchmark_id)], normalizer_id, metric_id, agg)


def ablation_rows(project: Project, runs: Sequence[Run], normalizer_id: str, metric_id: str, agg: str) -> Rows:
    """What each rule family of a normalizer is worth to each run: one row per run and family.

    ``normalizer_id`` is ``own`` or ``leaderboard_split``, the two that are made of families.
    ``change`` is the rise in the error rate, in points, when the family is switched off.
    The larger it is, the more that family helps the score. A run that is not scored yet
    under a family switched off is scored now, and saved.
    """
    out: Rows = []
    for run in runs:
        family_table = _run_ablation(project.root, project.stamp, run.system_id, run.benchmark_id, normalizer_id, metric_id, agg)
        if not family_table:
            continue  # semantic error, not computed for the families of this run
        every_family_on = family_table[0]  # the first row is the score itself, and the rest have a family off
        for row in family_table[1:]:
            out.append(
                {
                    "system": run.system_id,
                    "benchmark": run.benchmark_id,
                    "family": row["variant"].removeprefix("without "),
                    "change": row["change"],
                    "with every family on": every_family_on[agg],
                    "run_id": run.run_id,
                }
            )
    return out


# ── slices ───────────────────────────────────────────────────────────────────────────────


@st.cache_data(**CACHE)
def _run_slices(
    root: str,
    stamp: str,
    system_id: str,
    benchmark_id: str,
    normalizer_id: str,
    metric_id: str,
    by: str,
    bins: int,
    agg: str,
) -> Rows:
    paths, _, store = open_project(root)
    normalizer = normalizers(paths).get(normalizer_id)
    return slices.slice_run(store, Run(system_id, benchmark_id), normalizer, metric_id, by, bins, agg)


def slice_rows(
    project: Project,
    runs: Sequence[Run],
    normalizer_id: str,
    metric_id: str,
    by: str,
    bins: int,
    agg: str,
) -> tuple[Rows, list[str]]:
    """Each run's score per bin or value of a field, in rising order, and the runs with no such score yet."""
    rows: Rows = []
    missing = []
    for run in runs:
        try:
            sliced = _run_slices(
                project.root,
                project.stamp,
                run.system_id,
                run.benchmark_id,
                normalizer_id,
                metric_id,
                by,
                bins,
                agg,
            )
        except scoring.NotComputedError:
            missing.append(run.run_id)
            continue
        for row in sliced:
            rows.append(
                {
                    "system": run.system_id,
                    "benchmark": run.benchmark_id,
                    "value": row["value"],
                    "at": row["at"],
                    "score": row["score"],
                    "test_cases": row["test_cases"],
                    "reference units": row["ref_units"],
                }
            )
    return rows, missing


@st.cache_data(**CACHE)
def slice_examples(
    root: str,
    stamp: str,
    system_id: str,
    benchmark_id: str,
    normalizer_id: str,
    metric_id: str,
    by: str,
    bins: int,
    value: str,
    highest_first: bool,
    start: int,
) -> Rows:
    """Test cases of one bin or value of a run's slice, from place ``start``: each one's score and the two texts it compares."""
    paths, _, store = open_project(root)
    normalizer = normalizers(paths).get(normalizer_id)
    rows, _ = slices.examples(
        store,
        Run(system_id, benchmark_id),
        normalizer,
        metric_id,
        by,
        bins,
        value,
        highest_first,
        start,
    )
    return rows


# ── metadata fields ──────────────────────────────────────────────────────────────────────

# Fields that come from the reference text. On a set scored per session every chunk carries
# its whole session's reference, so these are counted once per session, not once per chunk.
OF_THE_REFERENCE = ("ref_words", "alphanumeric", "speaking_rate")


@st.cache_data(**CACHE)
def field_values(root: str, stamp: str, field_id: str, benchmark_id: str) -> list[Any]:
    """A field's value for every test case of a benchmark. A test case with no value for it gives None.

    For a field of the reference text on a set scored per session, such as Earnings22, there
    is one value per session.
    """
    _, _, store = open_project(root)
    test_cases = store.test_cases.read(benchmark_id)
    if field_id in OF_THE_REFERENCE and store.test_cases.scored_per_session(benchmark_id):
        # Keep each session's first chunk only.
        first_of_session = {}
        for test_case in test_cases:
            first_of_session.setdefault(test_case.parent_id, test_case)
        test_cases = list(first_of_session.values())
    return [getattr(test_case, field_id) for test_case in test_cases]


# ── semantic error, computed from the page ───────────────────────────────────────────────


def semantic_missing(project: Project, runs: Sequence[Run]) -> list[tuple[Run, str]]:
    """The run and normalizer pairs that have no saved semantic error, for sets scored per test case."""
    missing = []
    for run in runs:
        if project.store.test_cases.scored_per_session(run.benchmark_id):
            continue
        for normalizer_id in normalizer_ids(project):
            if not project.store.scores.has(run, normalizer_id, "semantic"):
                missing.append((run, normalizer_id))
    return missing


def compute_semantic(project: Project, pairs: Sequence[tuple[Run, str]], progress: Progress | None = None) -> int:
    """Compute and store semantic error for these run and normalizer pairs. Returns how many were done.

    It loads the sentence encoder and runs on this machine. The models on a benchmark are
    done together, so its references are embedded once.
    """
    registry = normalizers(project.paths)
    embedder = semantic_metric.SentenceTransformerEmbedder()

    together: dict[tuple[str, str], list[Run]] = {}  # (benchmark, normalizer) -> the runs to do together
    for run, normalizer_id in pairs:
        together.setdefault((run.benchmark_id, normalizer_id), []).append(run)

    done = 0
    for (_, normalizer_id), runs in together.items():
        if progress:
            progress(done, len(pairs))
        scoring.score_semantic(project.store, runs, registry.get(normalizer_id), embedder)
        done += len(runs)
    if progress:
        progress(len(pairs), len(pairs))
    return len(pairs)
