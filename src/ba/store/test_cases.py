"""A benchmark's test cases: ``test_cases/<benchmark>.parquet``.

The file is the benchmark's list, in order, with what describes each test case. It holds no
text: a test case's reference is with the transcriptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ba.store import parquet


@dataclass
class TestCase:
    """One item of a benchmark, and what describes it. Its reference is a text, and is with the transcriptions."""

    test_case_id: str
    parent_id: str | None = None  # the session a chunk belongs to
    chunk_index: int | None = None  # the chunk's place in its session
    audio_path: str | None = None  # relative to data/audio, once the audio is fetched
    # The metadata fields, each under its id: see ``ba.metadata``. A field that comes from
    # the audio is empty until the audio is fetched.
    duration: float | None = None
    snr_given: float | None = None
    sampling_rate: int | None = None
    ref_words: int | None = None
    alphanumeric: bool | None = None
    speaking_rate: float | None = None
    level: float | None = None
    snr_est: float | None = None
    bandwidth_est: float | None = None
    speech_fraction: float | None = None

    @property
    def unit_id(self) -> str:
        """The id of what is scored: this test case, or the session it is a chunk of.

        A session's id is its dataset, as a test case's id begins, and the session.
        """
        if self.parent_id is None:
            return self.test_case_id
        dataset = self.test_case_id.split("/", 1)[0]
        return f"{dataset}/{self.parent_id}"


# A column for each field of a test case.
_SCHEMA = pa.schema(
    [
        ("test_case_id", pa.string()),
        ("parent_id", pa.string()),
        ("chunk_index", pa.int32()),
        ("audio_path", pa.string()),
        ("duration", pa.float64()),
        ("snr_given", pa.float64()),
        ("sampling_rate", pa.int32()),
        ("ref_words", pa.int32()),
        ("alphanumeric", pa.bool_()),
        ("speaking_rate", pa.float64()),
        ("level", pa.float64()),
        ("snr_est", pa.float64()),
        ("bandwidth_est", pa.float64()),
        ("speech_fraction", pa.float64()),
    ]
)


class TestCaseTable:
    """Each benchmark's list of test cases: one file to a benchmark."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def _file(self, benchmark_id: str) -> Path:
        return self.folder / f"{benchmark_id}.parquet"

    def has(self, benchmark_id: str) -> bool:
        """Whether the benchmark's list is here."""
        return self._file(benchmark_id).exists()

    def write(self, benchmark_id: str, test_cases: list[TestCase]) -> None:
        """Keep a benchmark's test cases, in place of any that are here. Their order is the benchmark's list."""
        rows = [asdict(test_case) for test_case in test_cases]
        table = pa.Table.from_pylist(rows, schema=_SCHEMA)
        # What a score on the benchmark is of is kept in the file, beside the list.
        unit = "test_case"
        for test_case in test_cases:
            if test_case.parent_id:
                unit = "session"
        table = table.replace_schema_metadata({"unit": unit})
        parquet.write(table, self._file(benchmark_id))

    def _fetched_file(self, benchmark_id: str) -> Path:
        """The benchmark's file. A benchmark that has none yet is an error that says how to get it."""
        if not self.has(benchmark_id):
            raise KeyError(f"Benchmark '{benchmark_id}' has no test case list yet. Run: ba fetch --benchmark {benchmark_id}")
        return self._file(benchmark_id)

    def read(self, benchmark_id: str) -> list[TestCase]:
        """A benchmark's test cases, in list order."""
        return [TestCase(**row) for row in pq.read_table(self._fetched_file(benchmark_id)).to_pylist()]

    def ids(self, benchmark_id: str) -> list[str]:
        """The ids of a benchmark's test cases, in list order."""
        table = pq.read_table(self._fetched_file(benchmark_id), columns=["test_case_id"])
        return table.column("test_case_id").to_pylist()

    def scored_per_session(self, benchmark_id: str) -> bool:
        """Whether a score on this benchmark is of a whole session, as on Earnings22: its test cases are chunks of sessions."""
        metadata = pq.read_schema(self._fetched_file(benchmark_id)).metadata
        return metadata[b"unit"] == b"session"
