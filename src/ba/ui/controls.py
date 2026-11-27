"""The controls the pages share: what they offer, and what is chosen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import streamlit as st

from ba.aggregations.registry import AGGREGATIONS
from ba.metadata.derived import DERIVED
from ba.metadata.given import GIVEN
from ba.normalize.registry import normalizers
from ba.store.transcriptions import Run
from ba.ui import charts
from ba.ui.page import CACHE, Project, open_project

# What the metric control shows -> the metric's id.
METRICS = {"WER": "wer", "CER": "cer", "Semantic error": "semantic"}
# What the fields control shows -> the metadata fields of that kind, each with what it is.
SOURCES = {"Given": GIVEN, "Derived": DERIVED}


# ── what there is to choose from ─────────────────────────────────────────────────────────


@st.cache_data(**CACHE)
def _runs(root: str, stamp: str) -> list[Run]:
    _, cfg, store = open_project(root)
    system_ids = [system.id for system in cfg.systems()]
    listed = [benchmark.id for benchmark in cfg.benchmarks() if store.test_cases.has(benchmark.id)]
    return store.hypotheses.runs(system_ids, listed)


def runs(project: Project, benchmark_ids: Sequence[str] | None = None) -> list[Run]:
    """Each model's run on each benchmark, where its transcripts are here, in the config's order."""
    every = _runs(project.root, project.stamp)
    if benchmark_ids is None:
        return every
    return [run for run in every if run.benchmark_id in benchmark_ids]


def benchmarks_with_runs(project: Project) -> list[str]:
    """The benchmarks some model has a score on, in the config's order."""
    have = {run.benchmark_id for run in runs(project)}
    return [benchmark.id for benchmark in project.cfg.benchmarks() if benchmark.id in have]


def benchmarks_with_lists(project: Project) -> list[str]:
    """The benchmarks whose test case list is here, in the config's order."""
    return [benchmark.id for benchmark in project.cfg.benchmarks() if project.store.test_cases.has(benchmark.id)]


def per_test_case_benchmarks(project: Project) -> tuple[list[str], list[str]]:
    """The benchmarks whose runs can be broken down by test case, and those scored per session, which cannot."""
    per_session = set()
    for run in runs(project):
        if project.store.test_cases.scored_per_session(run.benchmark_id):
            per_session.add(run.benchmark_id)

    per_test_case = []
    not_per_test_case = []
    for benchmark_id in benchmarks_with_runs(project):
        if benchmark_id in per_session:
            not_per_test_case.append(benchmark_id)
        else:
            per_test_case.append(benchmark_id)
    return per_test_case, not_per_test_case


def models_with_runs(project: Project) -> list[str]:
    """The models that have a score somewhere, in the config's order."""
    have = {run.system_id for run in runs(project)}
    return [system.id for system in project.cfg.systems() if system.id in have]


def model_labels(project: Project) -> dict[str, str]:
    """What a chart's axis calls each model: its name without the organisation."""
    return charts.short_names([system.id for system in project.cfg.systems()])


def model_colours(project: Project) -> dict[str, str]:
    """Each model's colour, the same on every page."""
    return charts.model_colours([system.id for system in project.cfg.systems()])


def normalizer_ids(project: Project) -> list[str]:
    """The ids of the normalizers, from the text as written to the most forgiving."""
    return normalizers(project.paths).ids()


# ── the controls ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Choices:
    """What a page's shared controls are set to."""

    agg: str
    benchmark_ids: list[str]
    system_ids: list[str]
    metric_id: str | None = None  # on a page where the metric is picked
    normalizer_id: str | None = None  # on a page where the normalizer is picked


def _held(key: str, offered: Sequence[str]) -> None:
    """Forget a remembered choice that names something no longer offered."""
    held = st.session_state.get(key)
    if held is None:
        return
    for item in held:
        if item not in offered:
            del st.session_state[key]
            return


def pick_metric(column: Any, key: str) -> str:
    label = column.radio("Metric", list(METRICS), key=f"{key}_metric", horizontal=True)
    return METRICS[label]


def pick_normalizer(column: Any, project: Project, key: str) -> str:
    """The normalizer the scores are under. It opens on the leaderboard's."""
    offered = normalizer_ids(project)
    opens_on = 0
    if "leaderboard" in offered:
        opens_on = offered.index("leaderboard")
    return column.selectbox("Normalizer", offered, index=opens_on, key=f"{key}_normalizer")


def pick_aggregation(column: Any, key: str) -> str:
    """How the test cases' scores are put together. It opens on overall."""
    offered = AGGREGATIONS.ids()
    return column.radio(
        "Aggregation",
        offered,
        key=f"{key}_agg",
        horizontal=True,
        index=offered.index("overall"),
        help="mean and median are over test cases. overall is all errors over all reference words.",
    )


def pick_models(project: Project, key: str) -> list[str]:
    """The models that are switched on, in the config's order whatever order they were switched on in."""
    models = models_with_runs(project)
    labels = model_labels(project)
    _held(f"{key}_models", models)

    def label(model_id: str) -> str:
        return labels.get(model_id, model_id)

    switched_on = st.pills(
        "Models",
        models,
        selection_mode="multi",
        default=models,
        format_func=label,
        key=f"{key}_models",
        help="Switch a model off to leave it out of everything on this page.",
    )
    if not switched_on:
        return []
    return [model for model in models if model in switched_on]


def choose(project: Project, key: str, *, first: str = "metric") -> Choices:
    """The choices Results and Normalization share: a metric or a normalizer, the aggregation, the benchmarks and the models.

    ``first`` says what the first control picks: ``metric``, or ``normalizer`` on a page whose
    lines are the metrics. Every model can be switched off and on again.
    """
    offered = benchmarks_with_runs(project)
    one, two, three = st.columns([2, 2, 5])

    metric_id = None
    normalizer_id = None
    if first == "metric":
        metric_id = pick_metric(one, key)
    else:
        normalizer_id = pick_normalizer(one, project, key)

    agg = pick_aggregation(two, key)

    _held(f"{key}_benchmarks", offered)  # a benchmark that lost its transcripts is no longer offered
    benchmark_ids = three.multiselect("Benchmarks", offered, default=offered, key=f"{key}_benchmarks")

    system_ids = pick_models(project, key)
    return Choices(
        agg=agg,
        benchmark_ids=list(benchmark_ids),
        system_ids=system_ids,
        metric_id=metric_id,
        normalizer_id=normalizer_id,
    )


def chosen_runs(project: Project, choices: Choices) -> list[Run]:
    """The runs of the chosen models on the chosen benchmarks."""
    return [run for run in runs(project, choices.benchmark_ids) if run.system_id in choices.system_ids]
