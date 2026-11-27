"""The small registry of things chosen by id."""

from __future__ import annotations

import pytest

from ba.utils.registry import Registry


def test_a_registry_keeps_what_is_registered_in_order_and_names_what_exists():
    reg = Registry("normalizers")
    reg.register("leaderboard", "a")
    reg.register("own", "b")
    assert reg.ids() == ["leaderboard", "own"]
    assert reg.get("own") == "b" and "own" in reg
    with pytest.raises(KeyError, match="leaderboard"):
        reg.get("nope")
