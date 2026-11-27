"""How things are printed: tables that fit a terminal, and the short names and words in them.

Nothing here prints. Every function gives back text, for a command to print.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Collection, Iterable, Sequence
from typing import Any

from ba.utils.list_utils import unique

WIDTH = 80  # the usual width of a terminal: what is printed here is no wider, so the terminal does not break it
COLUMN_GAP = "  "  # between two columns of a table


def fill(text: str, width: int = WIDTH, indent: str = "") -> list[str]:
    """Prose as lines no wider than ``width``, broken at spaces only. Lines after the first start with ``indent``."""
    return textwrap.wrap(text, width, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False)


def _table_line(cells: list[str], widths: list[int], right: Collection[int]) -> str:
    """One line of a table: each cell padded to its column's width."""
    padded = []
    for column, cell in enumerate(cells):
        if column in right:
            padded.append(cell.rjust(widths[column]))
        else:
            padded.append(cell.ljust(widths[column]))
    return COLUMN_GAP.join(padded).rstrip()


def table(headers: list[str], rows: list[list[Any]], right: Collection[int] = ()) -> str:
    """Rows under their headings. The columns numbered in ``right`` hold figures, and are set to the right."""
    # Every cell as text.
    header_cells = [str(header) for header in headers]
    body = []
    for row in rows:
        body.append([str(cell) for cell in row])

    # Each column is as wide as its widest cell.
    widths = []
    for column in range(len(headers)):
        widest = len(header_cells[column])
        for row in body:
            widest = max(widest, len(row[column]))
        widths.append(widest)

    lines = [_table_line(header_cells, widths, right)]
    lines.append(COLUMN_GAP.join("-" * width for width in widths))
    for row in body:
        lines.append(_table_line(row, widths, right))
    return "\n".join(lines)


def error_message(error: BaseException) -> str:
    """What an error says. A KeyError's message is its key, without the quotes Python puts round it."""
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def plural(number: int, noun: str) -> str:
    """A count with its noun: 1 test case, 2 test cases."""
    if number == 1:
        return f"{number} {noun}"
    return f"{number} {noun}s"


def figure(value: float | None) -> str:
    """A score to two decimals, or a dash where there is none."""
    if value is None:
        return "-"
    return f"{value:.2f}"


def metric_name(metric_id: str) -> str:
    """WER and CER are initials. Another metric goes by its name: semantic error."""
    if metric_id in ("wer", "cer"):
        return metric_id.upper()
    return f"{metric_id} error"


def _name_starts(model_id: str) -> list[str]:
    """Every way to cut a model's name short at the end of a word, shortest first, and then the whole name.

    ``nvidia/NVIDIA-Nemotron-3.5`` gives ``NVIDIA``, ``NVIDIA-Nemotron`` and ``NVIDIA-Nemotron-3.5``.
    """
    name = model_id.split("/", 1)[-1]  # without the organisation
    starts = []
    for separator in re.finditer("[-_]", name):
        starts.append(name[: separator.start()])
    starts.append(name)
    return starts


def _first_words(starts: list[str], words: int) -> str:
    """A name's first ``words`` words, or the whole name if it has fewer."""
    index = min(words, len(starts)) - 1
    return starts[index]


def brief_names(model_ids: Sequence[str]) -> dict[str, str]:
    """A short name for each model, to stand for it in a table: an id is as wide as a table should be.

    It is the first words of the model's name, without its organisation: as many words as
    tell it from every other model. Two models with one name get their whole ids.
    """
    starts = {model_id: _name_starts(model_id) for model_id in model_ids}
    most_words = max((len(found) for found in starts.values()), default=0)

    brief = {}
    for model_id in model_ids:
        brief[model_id] = model_id  # the whole id, unless some number of its first words tells it apart
        for words in range(1, most_words + 1):
            mine = _first_words(starts[model_id], words)
            same_as_another = False
            for other_id in model_ids:
                if other_id != model_id and _first_words(starts[other_id], words) == mine:
                    same_as_another = True
            if not same_as_another:
                brief[model_id] = mine
                break
    return brief


def legend(brief: dict[str, str], model_ids: Iterable[str]) -> str:
    """The models a table names in short, and the id each name is short for."""
    rows = []
    for model_id in unique(model_ids):
        rows.append([brief[model_id], model_id])
    return table(["model", "short for"], rows)
