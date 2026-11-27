"""``ba reproduce``: each model's WER here beside the figures the leaderboard publishes."""

from __future__ import annotations

from typing import Any

import typer

from ba.actions.reproduce import reproduction
from ba.cli.common import (
    guard,
    setup,
    short_names,
)
from ba.normalize.registry import normalizers
from ba.utils.display_utils import figure, legend, plural, table
from ba.utils.list_utils import unique


def _reproduction_table(rows: list[dict[str, Any]], benchmark_id: str, brief: dict[str, str]) -> str:
    """One benchmark's table: each model's WER here beside the two figures the leaderboard publishes."""
    out = []
    for row in rows:
        if row["benchmark"] != benchmark_id:
            continue
        scored_of_test_cases = "-"
        if row["test_cases"] is not None:
            scored_of_test_cases = f"{row['scored']}/{row['test_cases']}"
        score_file_wer = None
        if row["score_file"] is not None:
            score_file_wer = row["score_file"]["wer"]
        out.append(
            [
                brief[row["system"]],
                scored_of_test_cases,
                figure(row["ours"]),
                figure(score_file_wer),
                figure(row["table"]),
                row["verdict"],
            ]
        )
    headers = ["model", "scored/test cases", "WER here", "its score file", "the table", "verdict"]
    return table(headers, out, right=[1, 2, 3, 4])


def _other_counts_table(rows: list[dict[str, Any]], brief: dict[str, str]) -> str:
    """The score files that reach our WER from other error counts: our counts and theirs, one under the other."""
    out = []
    for row in rows:
        name = brief[row["system"]]
        ours, theirs = row["counts"], row["score_file"]
        out.append([name, row["benchmark"], "here", ours["ins"], ours["del"], ours["sub"]])
        out.append([name, row["benchmark"], "its score file", theirs["ins"], theirs["del"], theirs["sub"]])
    return table(["model", "benchmark", "counts", "ins", "del", "sub"], out, right=[3, 4, 5])


def table_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each model's mean over the sets that have both our figure and the table's."""
    means = []
    for system in unique(row["system"] for row in rows):
        ours, theirs = [], []
        for row in rows:
            if row["system"] == system and row["ours"] is not None and row["table"] is not None:
                ours.append(row["ours"])
                theirs.append(row["table"])
        if ours:
            means.append(
                {
                    "system": system,
                    "sets": len(ours),
                    "ours": sum(ours) / len(ours),
                    "table": sum(theirs) / len(theirs),
                }
            )
    return means


def reproduce() -> None:
    """Score every model's transcripts again, and put each figure beside the leaderboard's own.

    Exits with 1 if a published WER is not the one the public transcripts give.
    """
    with guard():
        paths, cfg, store = setup()
        board = normalizers(paths).get("leaderboard")
        rows = reproduction(store, cfg, board, rescore=True)
        brief = short_names(cfg)

        typer.echo(legend(brief, [row["system"] for row in rows]))
        for benchmark_id in unique(row["benchmark"] for row in rows):
            typer.echo(f"\n{benchmark_id}")
            typer.echo(_reproduction_table(rows, benchmark_id, brief))

        checks = []  # one True or False for each published figure
        for row in rows:
            checks.extend(row["checks"])
        matched = checks.count(True)
        typer.echo(f"\n{matched} of {len(checks)} published figures matched.")

        other_counts = [row for row in rows if row["verdict"] == "counts differ"]
        if other_counts:
            verb = "has" if len(other_counts) == 1 else "have"
            typer.echo(f"\n{plural(len(other_counts), 'score file')} {verb} the same WER from other error counts:")
            typer.echo(_other_counts_table(other_counts, brief))

        means = table_means(rows)
        if means:
            mean_rows = []
            for mean in means:
                mean_rows.append([brief[mean["system"]], figure(mean["ours"]), figure(mean["table"]), mean["sets"]])
            typer.echo("\nMean WER over the sets the table shows:")
            typer.echo(table(["model", "here", "the table", "sets"], mean_rows, right=[1, 2, 3]))

        if matched != len(checks):
            raise typer.Exit(1)
