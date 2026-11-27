"""``ba ui``: open the web UI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from ba.cli.common import get_paths, guard


def ui(port: Annotated[int, typer.Option("--port")] = 8501) -> None:
    """Open the web UI: results, normalization, slices and benchmark metadata, read from what is stored."""
    with guard():
        try:
            import streamlit  # noqa: F401
        except ImportError as exc:
            raise ValueError("The web UI needs Streamlit. Install it with: uv sync --extra demo") from exc
        import ba.ui

        app_file = Path(ba.ui.__file__).parent / "app.py"
        command = [sys.executable, "-m", "streamlit", "run", str(app_file), "--server.port", str(port)]

        # The UI reads the project this command was run in.
        environment = dict(os.environ)
        environment["BA_ROOT"] = str(get_paths().root)

        exit_code = subprocess.call(command, env=environment)
        raise typer.Exit(exit_code)
