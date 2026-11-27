"""What every command shares: the project it works on, its errors, and how a model or a benchmark is named."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Annotated

import typer

from ba.config import Config, load_config
from ba.paths import Paths
from ba.store.store import Store
from ba.store.transcriptions import Run
from ba.utils.display_utils import brief_names, error_message

# The errors a command says in one line, without a traceback.
HANDLED = (KeyError, ValueError, OSError, RuntimeError)

NOTHING = "Nothing fetched yet. Run: ba fetch"

SystemOption = Annotated[
    list[str] | None,
    typer.Option("--system", help="A model's id, or the short name the tables show. Repeat for several. Default: every model in configs/systems.yaml."),
]

BenchmarkOption = Annotated[
    list[str] | None,
    typer.Option("--benchmark", help="A benchmark's id. Repeat for several. Default: every benchmark."),
]


def get_paths() -> Paths:
    return Paths.discover()


def setup() -> tuple[Paths, Config, Store]:
    """The project a command works on: where it is, its configuration, and what is stored in it."""
    paths = get_paths()
    cfg = load_config(paths)
    store = Store(paths)
    return paths, cfg, store


@contextmanager
def guard():
    """Turn an expected error into one line and exit code 1."""
    try:
        yield
    except (typer.Exit, typer.Abort):
        raise  # an exit code on purpose, not an error (both are RuntimeErrors underneath)
    except HANDLED as exc:
        if os.environ.get("BA_DEBUG"):
            raise  # the full traceback, for when the one line is not enough
        typer.echo(f"Error: {error_message(exc)}", err=True)
        raise typer.Exit(1) from exc


def short_names(cfg: Config) -> dict[str, str]:
    """The short name of every configured model: the same in every table, whichever models it shows."""
    model_ids = [system.id for system in cfg.systems()]
    return brief_names(model_ids)


def model_id(cfg: Config, name: str) -> str:
    """The id of the model that ``--system`` names: by its id, or by the short name the tables show, in any case."""
    brief = short_names(cfg)
    if name in brief:
        return name  # a whole id

    named = [model for model, short in brief.items() if short == name]
    if not named:
        # No short name is spelt exactly so. Any case will do, if it still names one model.
        named = [model for model, short in brief.items() if short.lower() == name.lower()]
    if len(named) != 1:
        raise KeyError(f"No system '{name}'. Configured: {', '.join(brief.values())}. A whole id works too: see configs/systems.yaml.")
    return named[0]


def picked(cfg: Config, systems: list[str] | None, benchmarks: list[str] | None) -> tuple[list[str], list[str]]:
    """The models and benchmarks a command covers: those named, else every one configured."""
    if systems:
        system_ids = [model_id(cfg, name) for name in systems]
    else:
        system_ids = [system.id for system in cfg.systems()]

    if benchmarks:
        benchmark_ids = [cfg.benchmark(name).id for name in benchmarks]
    else:
        benchmark_ids = [benchmark.id for benchmark in cfg.benchmarks()]

    return system_ids, benchmark_ids


def runs_of(store: Store, cfg: Config, systems: list[str] | None, benchmarks: list[str] | None) -> list[Run]:
    """Each chosen model's run on each chosen benchmark, where its transcripts are here."""
    system_ids, benchmark_ids = picked(cfg, systems, benchmarks)
    return store.hypotheses.runs(system_ids, benchmark_ids)
