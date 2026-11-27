"""``ba fetch``: read the public data. The fetching itself is in ``ba.fetching``."""

from __future__ import annotations

from typing import Annotated

import typer

from ba.cli.common import BenchmarkOption, SystemOption, guard, picked, setup
from ba.fetching import fetch as fetching


def fetch(
    system: SystemOption = None,
    benchmark: BenchmarkOption = None,
    audio: Annotated[
        bool,
        typer.Option("--audio", help="Also download the benchmarks' audio, to data/. Only the metadata fields measured on the audio need it."),
    ] = False,
) -> None:
    """Fetch each benchmark's test cases and each model's public transcripts.

    Public data, text only, no account is needed. A fetch starts over: everything stored is
    removed first, the scores with it, and everything is read again.
    """
    with guard():
        paths, cfg, store = setup()
        system_ids, benchmark_ids = picked(cfg, system, benchmark)
        fetching.fetch(paths, cfg, store, system_ids, benchmark_ids, audio=audio, say=typer.echo)
