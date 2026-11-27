"""The audio on disk: where a test case's file is, and what a file says of itself.

The audio is fetched only when asked for (``ba fetch --audio``). Nothing but the metadata
fields measured on it needs it.
"""

from __future__ import annotations

import io
from pathlib import Path

import soundfile as sf

from ba.paths import Paths
from ba.store.test_cases import TestCase


def probe(source: bytes | Path | str) -> tuple[int, str]:
    """``(sampling rate, container format)`` of an audio file, without decoding it.

    ``source`` is the file's bytes, or the path of a file on disk.
    """
    if isinstance(source, bytes):
        info = sf.info(io.BytesIO(source))
    else:
        info = sf.info(str(source))
    return info.samplerate, info.format


def audio_file(paths: Paths, test_case: TestCase) -> Path | None:
    """Where a test case's audio is on disk, or None if it has not been fetched."""
    if not test_case.audio_path:
        return None
    path = paths.audio / test_case.audio_path
    if not path.is_file():
        return None
    return path
