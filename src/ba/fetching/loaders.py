"""Reading a dataset's rows from the Parquet files the Hub stores.

``datasets`` is used once, to resolve a config and split to its files.
The rows are then read with pyarrow: whole files through the Hub's download cache when the
audio is wanted, or only the small columns over range requests when it is not.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ba.config import HfSource

# The clone's own fallbacks, in its order (normalizer/data_utils.py).
TEXT_COLUMNS = ("text", "sentence", "normalized_text", "transcript", "transcription")
ID_COLUMNS = ("id", "file_name")


def split_hf_url(url: str, repo: str) -> tuple[str | None, str]:
    """``(revision, path in repo)`` of an ``hf://datasets/<repo>@<revision>/<path>`` address.

    The revision may be left out of the address. It is then None.
    """
    rest = url.removeprefix(f"hf://datasets/{repo}")
    if rest.startswith("@"):
        revision, _, path = rest[1:].partition("/")
        return revision, path.lstrip("/")
    return None, rest.lstrip("/")


def _is_audio(column: pa.Field) -> bool:
    """The Hub stores a test case's audio as a struct that holds its bytes."""
    if not pa.types.is_struct(column.type):
        return False
    return "bytes" in [part.name for part in column.type]


@dataclass
class ParquetSource:
    """A dataset split as a list of Parquet files: local paths, or ``hf://`` addresses."""

    repo: str
    revision: str | None
    files: list[str]

    def _open(self, file: str, with_audio: bool) -> pq.ParquetFile:
        """Open one file. From the Hub, the whole file is downloaded only when the audio is wanted."""
        if not file.startswith("hf://"):
            return pq.ParquetFile(file)
        if with_audio:
            from huggingface_hub import hf_hub_download

            revision, path = split_hf_url(file, self.repo)
            local = hf_hub_download(
                self.repo,
                path,
                repo_type="dataset",
                revision=revision or self.revision,
                token=False,
            )
            return pq.ParquetFile(local)
        # Without the audio only the small columns are read, straight from the Hub.
        from huggingface_hub import HfFileSystem

        return pq.ParquetFile(HfFileSystem(token=False).open(file.removeprefix("hf://"), "rb"))

    def rows(self, with_audio: bool) -> Iterator[dict[str, Any]]:
        """Every row of the split, file by file. The audio column is left out unless it is wanted."""
        for file in self.files:
            parquet = self._open(file, with_audio)
            columns = []
            for column in parquet.schema_arrow:
                if with_audio or not _is_audio(column):
                    columns.append(column.name)
            for group in range(parquet.num_row_groups):
                yield from parquet.read_row_group(group, columns=columns).to_pylist()


def resolve_hf(source: HfSource) -> ParquetSource:
    """Resolve a Hub dataset config and split to its Parquet files, as the dataset is now.

    The sets are public, so no account is used. The files' addresses name the revision they
    are at, so every file of the split is read at that one revision.
    """
    from datasets import load_dataset_builder

    builder = load_dataset_builder(source.repo, source.config, token=False)
    files_by_split = builder.config.data_files or {}
    files = []
    for split, split_files in files_by_split.items():
        if str(split) == source.split:
            files.extend(str(file) for file in split_files)
    revision, _ = split_hf_url(files[0], source.repo)
    if revision is None:
        revision = getattr(builder, "hash", None)
    return ParquetSource(repo=source.repo, revision=revision, files=files)
