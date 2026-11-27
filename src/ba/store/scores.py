"""The scores: ``scores/ours/<model>__<benchmark>/<normalizer>.parquet``, and the leaderboard's own beside them.

Ours is one file for a run under a normalizer, with a row for each transcript. A row has a
group of columns for each metric. A metric's columns are blank until it is computed, and it
is computed for every row at once, so the file itself says which of its metrics are there.

A metric is written and read as rows of its own, the ones the aggregations read:
``test_case_id``, ``empty``, the counts ``n_ref``, ``n_sub``, ``n_del`` and ``n_ins``, and
``rate``. Word and character errors are counts. Semantic error is a rate.

The leaderboard's own score of a run's transcript file is in
``scores/huggingface/<model>__<benchmark>/leaderboard.parquet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ba.store import parquet
from ba.store.transcriptions import Run

_SCHEMA = pa.schema(
    [
        ("test_case_id", pa.string()),
        ("normalizer", pa.string()),
        ("empty", pa.bool_()),  # the model wrote nothing
        # Word errors: reference words, substitutions, deletions and insertions.
        ("wer_ref", pa.int64()),
        ("wer_sub", pa.int64()),
        ("wer_del", pa.int64()),
        ("wer_ins", pa.int64()),
        # Character errors, the same four.
        ("cer_ref", pa.int64()),
        ("cer_sub", pa.int64()),
        ("cer_del", pa.int64()),
        ("cer_ins", pa.int64()),
        # Semantic error: a rate, not a count.
        ("semantic", pa.float64()),
    ]
)

_PUBLISHED_SCHEMA = pa.schema(
    [
        ("wer", pa.float64()),
        ("wer_sub", pa.int64()),
        ("wer_del", pa.int64()),
        ("wer_ins", pa.int64()),
    ]
)


class ScoreTable:
    """Each run's scores under each normalizer: one folder to a run, one file to a normalizer."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _file(self, run: Run, normalizer_id: str) -> Path:
        return self.folder / run.run_id / f"{normalizer_id}.parquet"

    def has(self, run: Run, normalizer_id: str, metric_id: str) -> bool:
        """Whether a metric is computed for a run under a normalizer: its columns are there, and not blank."""
        path = self._file(run, normalizer_id)
        if not path.exists():
            return False
        column = "semantic" if metric_id == "semantic" else f"{metric_id}_ref"
        values = pq.read_table(path, columns=[column]).column(column).to_pylist()
        # A metric is computed for every row at once, so the first row tells whether it is there.
        return not values or values[0] is not None

    def write(self, run: Run, normalizer_id: str, metric_id: str, rows: list[dict[str, Any]]) -> None:
        """Keep one metric of a run under a normalizer, beside the metrics that are there already.

        ``rows`` are the metric's rows, one for each transcript. The first metric written
        makes the file's rows, and a later one fills in its own columns of them.
        """
        path = self._file(run, normalizer_id)
        kept = {}  # the file's rows, each under the id of what it is of
        if path.exists():
            for row in pq.read_table(path).to_pylist():
                kept[row["test_case_id"]] = row
        for row in rows:
            test_case_id = row["test_case_id"]
            if test_case_id not in kept:
                kept[test_case_id] = {"test_case_id": test_case_id, "normalizer": normalizer_id, "empty": row["empty"]}
            if metric_id == "semantic":
                kept[test_case_id]["semantic"] = row["rate"]
            else:
                kept[test_case_id][f"{metric_id}_ref"] = row["n_ref"]
                kept[test_case_id][f"{metric_id}_sub"] = row["n_sub"]
                kept[test_case_id][f"{metric_id}_del"] = row["n_del"]
                kept[test_case_id][f"{metric_id}_ins"] = row["n_ins"]
        parquet.write(pa.Table.from_pylist(list(kept.values()), schema=_SCHEMA), path)

    def read(self, run: Run, normalizer_id: str, metric_id: str) -> list[dict[str, Any]]:
        """One metric's rows for a run under a normalizer, one for each transcript, in list order.

        A row is what the aggregations read. For word and character errors it holds the
        counts. For semantic error it holds the ``rate``, and in ``n_ref`` its weight: the
        reference's word count, which the word errors beside it hold.
        """
        rows = []
        for kept in pq.read_table(self._file(run, normalizer_id)).to_pylist():
            row = {"test_case_id": kept["test_case_id"], "empty": kept["empty"]}
            if metric_id == "semantic":
                row["n_ref"] = kept["wer_ref"]
                row["n_sub"] = 0
                row["n_del"] = 0
                row["n_ins"] = 0
                row["rate"] = kept["semantic"]
            else:
                row["n_ref"] = kept[f"{metric_id}_ref"]
                row["n_sub"] = kept[f"{metric_id}_sub"]
                row["n_del"] = kept[f"{metric_id}_del"]
                row["n_ins"] = kept[f"{metric_id}_ins"]
                row["rate"] = None
            rows.append(row)
        return rows


class PublishedScoreTable:
    """The leaderboard's own score of each run's transcript file, where it published one."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _file(self, run: Run) -> Path:
        return self.folder / run.run_id / "leaderboard.parquet"

    def has(self, run: Run) -> bool:
        """Whether the leaderboard's own score of a run is here."""
        return self._file(run).exists()

    def write(self, run: Run, score: dict[str, Any]) -> None:
        """Keep the leaderboard's own score of a run: its WER and the three error counts behind it."""
        parquet.write(pa.Table.from_pylist([score], schema=_PUBLISHED_SCHEMA), self._file(run))

    def read(self, run: Run) -> dict[str, Any]:
        """The leaderboard's own score of a run: its WER and the three error counts behind it."""
        return pq.read_table(self._file(run)).to_pylist()[0]
