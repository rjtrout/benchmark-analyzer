"""The texts: ``transcriptions/references/<benchmark>/`` and ``transcriptions/hypotheses/<model>__<benchmark>/``.

A folder holds ``clean.parquet``, the texts as written, and a file for each normalizer that
has been applied to them: the same texts as that normalizer leaves them. Every file has two
columns: ``test_case_id``, the id of what is scored, and the text.

A benchmark's references are shared by every model on it. A model's transcripts of a
benchmark are a ``Run``. What a model has no transcript of has no row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ba.store import parquet

# The option of a text that no normalizer has touched: the references, or a run's transcripts, as written.
CLEAN = "clean"


@dataclass(frozen=True)
class Run:
    """One model's transcripts of one benchmark."""

    system_id: str
    benchmark_id: str

    @property
    def run_id(self) -> str:
        """What the run's files are named: the model, with its slash made a dash, and the benchmark."""
        return f"{self.system_id.replace('/', '-')}__{self.benchmark_id}"


_REFERENCES_SCHEMA = pa.schema(
    [
        ("test_case_id", pa.string()),
        ("reference", pa.string()),
    ]
)

_HYPOTHESES_SCHEMA = pa.schema(
    [
        ("test_case_id", pa.string()),
        ("hypothesis", pa.string()),
    ]
)


class ReferenceTable:
    """Each benchmark's references, by what is scored against each: one folder to a benchmark."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _file(self, benchmark_id: str, option: str) -> Path:
        return self.folder / benchmark_id / f"{option}.parquet"

    def has(self, benchmark_id: str, option: str = CLEAN) -> bool:
        """Whether a benchmark's references are here: as written, or as a normalizer leaves them."""
        return self._file(benchmark_id, option).exists()

    def write(self, benchmark_id: str, references: dict[str, str], option: str = CLEAN) -> None:
        """Keep a benchmark's references: as written, or under a normalizer's id as that normalizer leaves them."""
        parquet.write_texts(self._file(benchmark_id, option), references, _REFERENCES_SCHEMA)

    def read(self, benchmark_id: str, option: str = CLEAN) -> dict[str, str]:
        """A benchmark's references, in list order: as written, or as a normalizer leaves them."""
        return parquet.read_texts(self._file(benchmark_id, option))

    def count(self, benchmark_id: str) -> int:
        """How many references a benchmark has: its test cases, or its sessions where it is scored per session."""
        return pq.read_metadata(self._file(benchmark_id, CLEAN)).num_rows

    def forget(self, benchmark_id: str, normalizer_id: str) -> None:
        """Forget what a normalizer left of a benchmark's references. The references as written stay."""
        self._file(benchmark_id, normalizer_id).unlink(missing_ok=True)


class HypothesisTable:
    """Each run's transcripts, by what is scored: one folder to a run."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _file(self, run: Run, option: str) -> Path:
        return self.folder / run.run_id / f"{option}.parquet"

    def has(self, run: Run, option: str = CLEAN) -> bool:
        """Whether a run's transcripts are here: as written, or as a normalizer leaves them."""
        return self._file(run, option).exists()

    def runs(self, system_ids: list[str], benchmark_ids: list[str]) -> list[Run]:
        """These models' runs on these benchmarks, where there are transcripts, benchmark by benchmark."""
        found = []
        for benchmark_id in benchmark_ids:
            for system_id in system_ids:
                run = Run(system_id, benchmark_id)
                if self.has(run):
                    found.append(run)
        return found

    def write(self, run: Run, hypotheses: dict[str, str], option: str = CLEAN) -> None:
        """Keep a run's transcripts: as written, or under a normalizer's id as that normalizer leaves them."""
        parquet.write_texts(self._file(run, option), hypotheses, _HYPOTHESES_SCHEMA)

    def read(self, run: Run, option: str = CLEAN) -> dict[str, str]:
        """A run's transcripts, in list order: as written, or as a normalizer leaves them."""
        return parquet.read_texts(self._file(run, option))

    def forget(self, run: Run, normalizer_id: str) -> None:
        """Forget what a normalizer left of a run's transcripts. The transcripts as written stay."""
        self._file(run, normalizer_id).unlink(missing_ok=True)
