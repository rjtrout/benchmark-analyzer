"""The one config object: the models compared, and the benchmarks.

Loaded from ``configs/systems.yaml`` and ``configs/benchmarks.yaml``. Nothing here chooses
what is run: every normalizer, metric and aggregation there is gets run on all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ba.paths import Paths


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SystemConfig(_Model):
    """A model whose transcripts the leaderboard publishes."""

    id: str  # as the leaderboard names it: the model's repo on the Hub
    published: dict[str, float] = Field(default_factory=dict)  # benchmark id -> the table's WER


class HfSource(_Model):
    repo: str
    config: str | None = None
    split: str = "test"


class ChunkedSpec(_Model):
    """A set stored as audio chunks with one reference per session, kept in a second repo."""

    metadata_file: str  # the index of chunks: id, parent_id, chunk_index, file_name
    audio_dir: str = "audio"
    parent_repo: str
    parent_metadata_file: str  # one row per session: id and its transcript
    text_column: str = "transcript"


class BenchmarkConfig(_Model):
    id: str
    loader: Literal["hf", "hf-chunked"]
    dataset_key: str
    source: HfSource
    chunked: ChunkedSpec | None = None
    manifest_aliases: list[str] = Field(default_factory=list)  # older names its public transcripts go by

    @property
    def manifest_dataset_ids(self) -> list[str]:
        """The dataset part of this benchmark's manifest file names: the current one, then older ones.

        The leaderboard names a file after the repo, the config and the split: ``org-repo_config_split``.
        """
        repo = self.source.repo.replace("/", "-")
        config = self.source.config or ""
        current = f"{repo}_{config}_{self.source.split}"
        return [current, *self.manifest_aliases]


class Config:
    def __init__(
        self,
        systems: list[SystemConfig],
        benchmarks: list[BenchmarkConfig],
    ) -> None:
        self._systems = {system.id: system for system in systems}
        self._benchmarks = {benchmark.id: benchmark for benchmark in benchmarks}

    def benchmark(self, id: str) -> BenchmarkConfig:
        if id not in self._benchmarks:
            raise KeyError(f"No benchmark '{id}'. Configured: {', '.join(self._benchmarks)}")
        return self._benchmarks[id]

    def systems(self) -> list[SystemConfig]:
        return list(self._systems.values())

    def benchmarks(self) -> list[BenchmarkConfig]:
        return list(self._benchmarks.values())


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(paths: Paths) -> Config:
    """Read the two config files into one ``Config``. Pydantic checks each entry's shape."""
    systems_file = _read_yaml(paths.configs / "systems.yaml")
    benchmarks_file = _read_yaml(paths.configs / "benchmarks.yaml")

    systems = [SystemConfig(**entry) for entry in systems_file["systems"]]
    benchmarks = [BenchmarkConfig(**entry) for entry in benchmarks_file["benchmarks"]]
    return Config(systems, benchmarks)
