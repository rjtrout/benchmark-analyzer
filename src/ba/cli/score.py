"""``ba score``: the score tables, with every count beside the number."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from ba.actions import score_summary
from ba.aggregations.registry import AGGREGATIONS
from ba.cli.common import (
    NOTHING,
    BenchmarkOption,
    SystemOption,
    guard,
    runs_of,
    setup,
    short_names,
)
from ba.metrics.registry import METRICS
from ba.normalize.own import FAMILIES
from ba.normalize.registry import FAMILIES_OF, normalizers, with_families
from ba.utils.display_utils import legend, metric_name, plural, table
from ba.utils.list_utils import unique


def print_scores(
    rows: list[dict[str, Any]],
    aggregation_ids: list[str],
    brief: dict[str, str],
    per_session: set[str],
) -> None:
    """A table for each benchmark and metric: a row for each model and normalizer, with every count beside the number.

    What every row of a table shares is said once, above it: the benchmark, the metric,
    how many test cases the benchmark has (or sessions, if it is scored per session), and the
    normalizer when there is only one. With several, the normalizer is a column.
    """
    if not rows:
        typer.echo(NOTHING)
        return
    typer.echo(legend(brief, [row["system"] for row in rows]))

    normalizer_ids = unique(row["normalizer"] for row in rows)
    several_normalizers = len(normalizer_ids) > 1
    if several_normalizers:
        headers = ["model", "normalizer", "scored", "ref units", "empty", *aggregation_ids]
        first_figure = 2
    else:
        headers = ["model", "scored", "ref units", "empty", *aggregation_ids]
        first_figure = 1

    for benchmark_id, metric_id in unique((row["benchmark"], row["metric"]) for row in rows):
        group = [row for row in rows if row["benchmark"] == benchmark_id and row["metric"] == metric_id]
        out = []
        for row in group:
            cells = [brief[row["system"]]]
            if several_normalizers:
                cells.append(row["normalizer"])
            cells += [row["scored"], row["ref_units"], row["empty"]]
            for aggregation_id in aggregation_ids:
                cells.append(f"{row[aggregation_id]:.2f}")
            out.append(cells)

        said = [metric_name(metric_id)]
        if not several_normalizers:
            said.append(f"{normalizer_ids[0]} normalizer")
        unit = "session" if benchmark_id in per_session else "test case"
        said.append(plural(group[0]["test_cases"], unit))
        typer.echo(f"\n{benchmark_id}: {', '.join(said)}")
        typer.echo(table(headers, out, right=range(first_figure, len(headers))))


def score(
    system: SystemOption = None,
    benchmark: BenchmarkOption = None,
    normalizer: Annotated[
        list[str] | None,
        typer.Option("--normalizer", help="Repeat for several. Default: every normalizer."),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option("--metric", help="wer, cer or semantic. Repeat for several. Default: wer."),
    ] = None,
    agg: Annotated[
        list[str] | None,
        typer.Option("--agg", help="mean, median or overall. Repeat for several. Default: all three."),
    ] = None,
    without: Annotated[
        list[str] | None,
        typer.Option(
            "--without",
            help=f"With --normalizer own or leaderboard_split: switch a rule family off. Repeat for several. The families of own: {', '.join(FAMILIES)}. The last four are not in leaderboard_split.",
        ),
    ] = None,
    rescore: Annotated[
        bool,
        typer.Option("--rescore", help="Score again from the transcripts as written, even where a score is saved."),
    ] = False,
) -> None:
    """Score the models' transcripts, with every count beside the number."""
    with guard():
        paths, cfg, store = setup()

        registry = normalizers(paths)
        normalizer_ids = normalizer or registry.ids()
        if without:
            if len(normalizer_ids) != 1 or normalizer_ids[0] not in FAMILIES_OF:
                raise ValueError("Only own and leaderboard_split have families to switch off. Use --normalizer own, or --normalizer leaderboard_split.")
            chosen_normalizers = [with_families(paths, normalizer_ids[0], without=without)]
        else:
            chosen_normalizers = [registry.get(normalizer_id) for normalizer_id in normalizer_ids]

        # An unknown aggregation or metric is an error before anything is scored.
        aggregation_ids = agg or AGGREGATIONS.ids()
        for aggregation_id in aggregation_ids:
            AGGREGATIONS.get(aggregation_id)
        metric_ids = metric or ["wer"]
        for metric_id in metric_ids:
            METRICS.get(metric_id)

        runs = runs_of(store, cfg, system, benchmark)
        per_session = set()  # the benchmarks whose scores are of whole sessions
        for run in runs:
            if store.test_cases.scored_per_session(run.benchmark_id):
                per_session.add(run.benchmark_id)
        rows = score_summary.score_table(
            store,
            runs,
            chosen_normalizers,
            aggregation_ids,
            rescore=rescore,
            metric_ids=metric_ids,
        )
        print_scores(rows, aggregation_ids, short_names(cfg), per_session)
        if "semantic" in metric_ids:
            typer.echo("\nSemantic error is shown where it has been computed. Compute it with: ba semantic")
