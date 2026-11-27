"""Small helpers for lists."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def unique(items: Iterable[Any]) -> list[Any]:
    """Each item once, in the order it first appears."""
    return list(dict.fromkeys(items))
