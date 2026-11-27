"""The leaderboard's own English normalizer, read from its code, which is kept as a git submodule.

The submodule is the leaderboard's published code and nothing here patches it. Which
commit it is at is the reader's to manage, with git.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from ba.paths import Paths


def ensure_importable(paths: Paths) -> Path:
    """Put the clone on ``sys.path`` so that its ``normalizer`` package imports."""
    root = paths.clone
    if not (root / "normalizer" / "normalizer.py").is_file():
        raise RuntimeError(f"The leaderboard's code is not at {root}. Run: git submodule update --init")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@lru_cache(maxsize=8)
def _english_normalizer(root: str):
    from normalizer.normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer()


def english_normalizer(paths: Paths):
    """The leaderboard's English normalizer, built the way its ``data_utils`` builds it."""
    return _english_normalizer(str(ensure_importable(paths)))

