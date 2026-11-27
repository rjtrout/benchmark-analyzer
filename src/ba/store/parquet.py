"""How the store's files are written and read: Parquet, as small as it makes them."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


def write(table: pa.Table, path: Path) -> None:
    """Keep a table as a file: as small as Parquet makes it, and whole or not at all.

    These files are what a commit carries, so they are written as small as Parquet makes
    them. That takes a moment, and a page of the web UI or another command may read the
    store meanwhile. So the table is written under a name of its own first, and takes the
    file's name only once it is whole. A reader finds the file as it was, or no file yet,
    and never half of one. Two runs may write the same file at once, so each writes under
    a different name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    unfinished = path.with_name(f"{path.name}.{uuid4().hex}.part")
    try:
        pq.write_table(table, unfinished, compression="zstd", compression_level=19)
        unfinished.replace(path)
    finally:
        # Nothing stays under that name: it has the file's name by now, or the write was stopped.
        unfinished.unlink(missing_ok=True)


def write_texts(path: Path, texts: dict[str, str], schema: pa.Schema) -> None:
    """A file of texts, each under the id of what is scored. The schema names the two columns."""
    id_column, text_column = schema.names
    table = pa.table({id_column: list(texts), text_column: list(texts.values())}, schema=schema)
    write(table, path)


def read_texts(path: Path) -> dict[str, str]:
    """A file of texts, by the id of what is scored, in the file's order."""
    table = pq.read_table(path)
    return dict(zip(table.column(0).to_pylist(), table.column(1).to_pylist(), strict=True))
