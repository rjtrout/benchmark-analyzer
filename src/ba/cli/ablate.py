"""``ba ablate``: what each rule family of a normalizer is worth."""

from __future__ import annotations

from typing import Annotated

import typer

from ba.actions import ablation
from ba.cli.common import (
    NOTHING,
    BenchmarkOption,
    SystemOption,
    guard,
    runs_of,
    setup,
    short_names,
)
from ba.utils.display_utils import legend, metric_name, table
from ba.utils.list_utils import unique


def ablate(
    system: SystemOption = None,
    benchmark: BenchmarkOption = None,
    normalizer: Annotated[
        str,
        typer.Option("--normalizer", help="own, or leaderboard_split for the leaderboard's rules alone."),
    ] = "own",
    metric: Annotated[str, typer.Option("--metric", help="wer, cer or semantic.")] = "wer",
    agg: Annotated[str, typer.Option("--agg")] = "overall",
    recompute: Annotated[
        bool,
        typer.Option("--recompute", help="Work it out again from the transcripts as written, even where it is saved."),
    ] = False,
) -> None:
    """Switch each rule family of a normalizer off in turn and show how the score moves.

    The own normalizer is the leaderboard's rules and four more of ours. A family switched
    off is a normalizer like any other. Its scores are counted once for each model and
    benchmark, and saved. After that they are read, for any aggregation. Semantic error is
    read too, once `ba semantic --families` has computed it.
    """
    with guard():
        _, cfg, store = setup()
        runs = runs_of(store, cfg, system, benchmark)
        if not runs:
            typer.echo(NOTHING)
            return
        family_table = ablation.ablate(store, runs, normalizer, metric, agg, recompute=recompute)
        if not family_table:
            # Word and character errors are counted as they are asked for, so it is semantic error that is not there.
            typer.echo(f"Semantic error is not computed for the rule families of {normalizer} yet. Compute it with: ba semantic --normalizer {normalizer} --families")
            return
        # A run with no row is left out: a set scored per session has no semantic error.
        in_table = {row["run_id"] for row in family_table}
        runs = [run for run in runs if run.run_id in in_table]
        brief = short_names(cfg)

        # Each run's row under each variant, and the variants in the table's order.
        row_of = {}
        for row in family_table:
            row_of[(row["run_id"], row["variant"])] = row
        variants = unique(row["variant"] for row in family_table)

        typer.echo(f"What each rule family of the {normalizer} normalizer is worth.")
        typer.echo("First row: the score. Other rows: the change in points when that family is off.")
        typer.echo("\n" + legend(brief, [run.system_id for run in runs]))
        for benchmark_id in unique(run.benchmark_id for run in runs):
            runs_here = [run for run in runs if run.benchmark_id == benchmark_id]
            headers = ["variant"] + [brief[run.system_id] for run in runs_here]
            rows = []
            for index, variant in enumerate(variants):
                cells = [variant]
                for run in runs_here:
                    row = row_of[(run.run_id, variant)]
                    if index == 0:
                        cells.append(f"{row[agg]:.2f}")  # the score itself
                    else:
                        cells.append(f"{row['change']:+.2f}")  # the change from the first row
                rows.append(cells)
            typer.echo(f"\n{benchmark_id}: {agg} {metric_name(metric)}")
            typer.echo(table(headers, rows, right=range(1, len(headers))))
