"""Clear one category of stored artifacts or downloads."""

from __future__ import annotations

from typing import Annotated

import typer

from ba.cli.common import guard, setup
from ba.store.store import CLEARABLE


def clear(what: Annotated[str, typer.Argument(help=" | ".join(CLEARABLE))]) -> None:
    """Remove one category of saved files after confirmation.

    `scores` removes normalized text and locally computed scores. `leaderboard-scores`
    removes imported published scores. `test-cases` removes saved test cases and metadata.
    `audio` and `transcript-files` remove the corresponding downloads.
    """
    with guard():
        if what not in CLEARABLE:
            raise ValueError(f"Choose one of: {', '.join(CLEARABLE)}")
        _, _, store = setup()
        typer.confirm(f"Are you sure you want to remove {what}?", abort=True)
        store.clear(what)
        typer.echo(f"Removed: {what}")
