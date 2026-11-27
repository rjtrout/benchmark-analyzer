"""One small registry of things chosen by id: normalizers, metrics and aggregations."""

from __future__ import annotations

from typing import Any


class Registry:
    """Things kept under an id, in the order they were registered."""

    def __init__(self, kind: str) -> None:
        self.kind = kind  # what it holds, for the message when an id is not found
        self._items: dict[str, Any] = {}

    def register(self, id: str, item: Any) -> None:
        self._items[id] = item

    def get(self, id: str) -> Any:
        """The item with this id. An unknown id is an error that names the ids there are."""
        if id not in self._items:
            known = ", ".join(self._items) or "nothing"
            raise KeyError(f"No {self.kind} entry '{id}'. Registered: {known}")
        return self._items[id]

    def ids(self) -> list[str]:
        return list(self._items)

    def __contains__(self, id: str) -> bool:
        return id in self._items
