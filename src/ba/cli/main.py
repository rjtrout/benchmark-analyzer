"""ba: compare the open models of the Open ASR Leaderboard from the transcripts it publishes.

Nothing is transcribed here. `ba fetch` reads public data, with no account, and every
other command works on what it stored.
"""

import typer

from ba.cli import (
    ablate,
    clear,
    fetch,
    metadata,
    reproduce,
    score,
    semantic,
    slice,
    test_metadata,
    ui,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

# The commands, in the order `ba --help` lists them. Each is in the module that has its name.
app.command()(fetch.fetch)
app.command()(reproduce.reproduce)
app.command()(score.score)
app.command()(ablate.ablate)
app.command()(semantic.semantic)
app.command()(metadata.metadata)
app.command("test_metadata")(test_metadata.test_metadata)  # named here, or Typer writes it test-metadata
app.command("slice")(slice.slice_)  # `slice` alone is Python's own name for part of a sequence
app.command()(clear.clear)
app.command()(ui.ui)

if __name__ == "__main__":
    app()
