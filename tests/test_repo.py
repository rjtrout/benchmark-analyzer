"""The repository itself: what a commit would leave out."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.conftest import REPO_ROOT


def test_no_source_test_or_config_file_is_hidden_by_the_ignore_rules():
    # An ignore rule for a folder at the root, such as data/, also matches a folder of the
    # same name deeper down. A commit would then leave that folder out, and a clone would not start.
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = [
        str(path.relative_to(REPO_ROOT))
        for folder in ("src", "tests", "configs", "artifacts")
        for path in (REPO_ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.name != ".DS_Store"
    ]  # fmt: skip
    asked = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"], input="\n".join(files), capture_output=True, text=True, cwd=REPO_ROOT
    )  # fmt: skip
    assert asked.returncode in (0, 1), asked.stderr
    assert asked.stdout.split() == []
