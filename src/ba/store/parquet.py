"""How the store's files are written and read: Parquet, as small as it makes them."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write(table: pa.Table, path: Path) -> None:
    """These files are what a commit carries, so they are written as small as Parquet makes them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=19)


def write_texts(path: Path, texts: dict[str, str], schema: pa.Schema) -> None:
    """A file of texts, each under the id of what is scored. The schema names the two columns."""
    id_column, text_column = schema.names
    table = pa.table({id_column: list(texts), text_column: list(texts.values())}, schema=schema)
    write(table, path)


def read_texts(path: Path) -> dict[str, str]:
    """A file of texts, by the id of what is scored, in the file's order."""
    table = pq.read_table(path)
    return dict(zip(table.column(0).to_pylist(), table.column(1).to_pylist(), strict=True))
