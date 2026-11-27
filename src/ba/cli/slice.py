"""``ba slice``: a score broken down by a metadata field."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from ba.actions import scoring, slices
from ba.cli.common import BenchmarkOption, SystemOption, guard, runs_of, setup, short_names
from ba.normalize.registry import normalizers
from ba.utils.display_utils import fill, legend, metric_name, table
from ba.utils.list_utils import unique


def slice_(
    by: Annotated[
        str,
        typer.Option("--by", help=f"The metadata field to break down by: {', '.join(slices.SLICES)}."),
    ],
    system: SystemOption = None,
    benchmark: BenchmarkOption = None,
    metric: Annotated[str, typer.Option("--metric", help="wer, cer or semantic.")] = "wer",
    agg: Annotated[str, typer.Option("--agg")] = "overall",
    bins: Annotated[
        int,
        typer.Option("--bins", help="For a number: how many bins, each holding the same number of test cases."),
    ] = slices.BINS,
    normalizer: Annotated[str, typer.Option("--normalizer")] = "leaderboard",
) -> None:
    """Break a score down by a metadata field. Every row shows the test cases and words behind it."""
    with guard():
        paths, cfg, store = setup()
        chosen = normalizers(paths).get(normalizer)
        brief = short_names(cfg)

        tables: dict[str, list[list[Any]]] = {}  # each benchmark's rows
        shown = []  # the models with a row in some table
        left_out = []  # what has nothing to break down, and why
        for run in runs_of(store, cfg, system, benchmark):
            if store.test_cases.scored_per_session(run.benchmark_id):
                left_out.append(run.benchmark_id)  # scored per session: there is no test case to slice by
                continue
            try:
                sliced = slices.slice_run(store, run, chosen, metric, by, bins, agg)
            except scoring.NotComputedError:
                left_out.append(f"{run.benchmark_id} (no {metric} yet for {run.system_id})")
                continue
            if not sliced:
                continue
            rows = tables.setdefault(run.benchmark_id, [])
            for row in sliced:
                score = f"{row['score']:.2f}"
                rows.append([brief[run.system_id], row["value"], row["test_cases"], row["ref_units"], score])
            shown.append(run.system_id)

        typer.echo(f"{agg} {metric_name(metric)} by {by}, {normalizer} normalizer")
        if tables:
            typer.echo("\n" + legend(brief, shown))
        else:
            typer.echo("No model has a score to break down here.")
        for benchmark_id, rows in tables.items():
            typer.echo(f"\n{benchmark_id}")
            typer.echo(table(["model", by, "test cases", "ref units", agg], rows, right=[2, 3, 4]))
        if left_out:
            note = f"Left out: {', '.join(unique(left_out))}. A set scored per session has no test case to slice by."
            typer.echo("\n" + "\n".join(fill(note)))
