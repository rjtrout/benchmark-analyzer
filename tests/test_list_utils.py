"""Small helpers for lists."""

from __future__ import annotations

from pathlib import Path

from ba.utils.list_utils import unique


def test_unique_keeps_each_item_once_in_the_order_it_first_appears():
    assert unique(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert unique([]) == []
    # Anything that can be looked up: pairs, paths, and what a loop hands over one at a time.
    assert unique([("ami", "wer"), ("ami", "cer"), ("ami", "wer")]) == [("ami", "wer"), ("ami", "cer")]
    assert unique(Path(name).parent for name in ["x/1", "y/2", "x/3"]) == [Path("x"), Path("y")]


