"""Where things live. Every other module takes a ``Paths`` instead of guessing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "benchmark-analyzer"


def find_root(start: Path | None = None) -> Path:
    """The project root: ``$BA_ROOT``, else the nearest parent holding our pyproject.toml."""
    env = os.environ.get("BA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        pyproject = folder / "pyproject.toml"
        if pyproject.is_file() and f'name = "{PROJECT_NAME}"' in pyproject.read_text(encoding="utf-8"):
            return folder
    # Installed in editable mode and run from elsewhere: src/ba/paths.py -> the repo.
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    configs: Path | None = None
    clone: Path | None = None  # the leaderboard's code, a git submodule
    urgent_clone: Path | None = None  # the URGENT authors' analysis code, a git submodule

    def __post_init__(self) -> None:
        # The folders that were not given are the ones under the root. The class is frozen so
        # that two Paths of the same folders are equal, and one stands in for the other where
        # something is remembered by its paths. A frozen class sets its fields this way.
        root = Path(self.root)
        if self.configs:
            configs = Path(self.configs)
        else:
            configs = root / "configs"
        if self.clone:
            clone = Path(self.clone)
        else:
            clone = root / "third_party" / "open_asr_leaderboard"
        if self.urgent_clone:
            urgent_clone = Path(self.urgent_clone)
        else:
            urgent_clone = root / "third_party" / "urgent2024_analysis"
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "configs", configs)
        object.__setattr__(self, "clone", clone)
        object.__setattr__(self, "urgent_clone", urgent_clone)

    @classmethod
    def discover(cls, start: Path | None = None) -> Paths:
        return cls(root=find_root(start))

    @property
    def artifacts(self) -> Path:
        """What the comparison is made of and its results: test case lists, transcripts, scores. Text only."""
        return self.root / "artifacts"

    @property
    def data(self) -> Path:
        """Git-ignored local cache: the downloaded transcript files, and audio if it is fetched."""
        return self.root / "data"

    @property
    def audio(self) -> Path:
        return self.data / "audio"

    @property
    def leaderboard(self) -> Path:
        return self.data / "leaderboard"
