"""A page: the project it reads, the frame every page starts with, and the small things every page draws.

Every page starts with ``frame()``. Everything slow is behind ``st.cache_data`` and keyed by
the project's root and a stamp of its files, so newly fetched transcripts or a new list are
seen at the next click. Nothing is transcribed here: every score is of stored transcripts.
"""

from __future__ import annotations

import hashlib
import html
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ba.config import Config, load_config
from ba.paths import Paths
from ba.store.store import Store
from ba.utils.display_utils import error_message

# Files under artifacts/ whose change means a cached read is out of date.
_WATCHED = (
    "test_cases/*.parquet",
    "transcriptions/hypotheses/*/clean.parquet",
    "scores/ours/*/*.parquet",
)

# How st.cache_data keeps every cached read.
CACHE: dict[str, Any] = {"show_spinner": False, "max_entries": 256}


@dataclass(frozen=True)
class Project:
    paths: Paths
    cfg: Config
    store: Store
    stamp: str

    @property
    def root(self) -> str:
        return str(self.paths.root)


def open_project(root: str | Path | None = None) -> tuple[Paths, Config, Store]:
    """The paths, config and store, as ``ba``'s own setup makes them."""
    if root is None:
        paths = Paths.discover()
    else:
        paths = Paths(root=Path(root))
    return paths, load_config(paths), Store(paths)


def store_stamp(paths: Paths) -> str:
    """A short hash that changes when the transcripts, a list, a stored field or the config changes."""
    files = list(paths.configs.glob("*.yaml"))
    for pattern in _WATCHED:
        files.extend(paths.artifacts.glob(pattern))

    parts = []
    for file in sorted(files):
        try:
            info = file.stat()
        except OSError:
            continue  # removed since it was listed
        parts.append(f"{file}:{info.st_mtime_ns}:{info.st_size}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def frame() -> Project:
    """Draw the sidebar every page shares, and return the project."""
    paths, cfg, store = open_project()
    st.sidebar.caption(f"Reading {paths.root}. Every number comes from transcripts the Open ASR Leaderboard publishes. Nothing is transcribed here.")
    return Project(paths, cfg, store, stamp=store_stamp(paths))


@contextmanager
def guard() -> Iterator[None]:
    """Show an expected error as one line, as the command line does. ``BA_DEBUG=1`` shows the traceback."""
    try:
        yield
    except (KeyError, ValueError, RuntimeError) as exc:
        if os.environ.get("BA_DEBUG"):
            raise
        st.error(error_message(exc))


def is_dark() -> bool:
    """Whether the page is drawn dark. False where there is no page to ask, as in a test."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def legend(colours: dict[str, str]) -> None:
    """One legend for every chart on a page: a square of each name's colour, and the name."""
    square = "display:inline-block;width:0.8em;height:0.8em;border-radius:2px"
    chips = []
    for name, colour in colours.items():
        chips.append(f'<span style="{square};background:{colour}"></span> {html.escape(name)}')
    st.html(f'<div style="font-size:0.85rem;line-height:1.9">{" &nbsp; ".join(chips)}</div>')


def wrapped_table(rows: Sequence[dict[str, Any]]) -> None:
    """A table to read: every cell is shown whole, on as many lines as it takes."""
    st.table(pd.DataFrame(list(rows)))


def table(rows: Sequence[dict[str, Any]], *, decimals: Sequence[str] = ()) -> None:
    """A table of rows. ``decimals`` columns show two decimals."""
    data = pd.DataFrame(list(rows))
    column_config = {}
    for column in decimals:
        if column in data.columns:
            column_config[column] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(data, hide_index=True, column_config=column_config)
