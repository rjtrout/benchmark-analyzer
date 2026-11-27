"""Filling in the metadata fields: what a test case's source states, then what is derived here.

``populate_metadata`` fills one test case. ``populate_speaking_rate`` fills the speaking
rate over a benchmark's whole list, since a session's chunks share theirs.
"""

from __future__ import annotations

from typing import Any

from ba.metadata.derived import get_derived
from ba.metadata.given import get_given
from ba.paths import Paths
from ba.store.test_cases import TestCase
from ba.utils.audio_utils import audio_file


def populate_metadata(test_case: TestCase, row: dict[str, Any], reference: str, paths: Paths) -> None:
    """Fill in a test case's metadata fields: first what its source states, then what is derived here.

    ``row`` is the test case's row of its dataset, and ``reference`` its reference as written.
    The fields that come from the audio are filled once the audio is on disk, and are empty
    until then. The speaking rate is left to ``populate_speaking_rate``.
    """
    audio = audio_file(paths, test_case)
    metadata = {}
    metadata.update(get_given(row, audio))
    metadata.update(get_derived(reference, audio, paths))
    for field, value in metadata.items():
        setattr(test_case, field, value)


def populate_speaking_rate(test_cases: list[TestCase]) -> None:
    """Fill in the speaking rate of a benchmark's test cases: the reference's words over the seconds they were said in.

    It is filled over the whole list, after ``populate_metadata`` has counted the words. A
    test case scored on its own has its own words over its own length. A chunk of a session
    carries its whole session's reference, so the words it holds are the session's: its rate
    is the session's, those words over the length of the session's chunks together. Pauses
    count as length. Where no length is stated, the rate is left empty.
    """
    seconds_of: dict[str, float] = {}  # what is scored -> its length: a test case's own, or its session's chunks together
    for test_case in test_cases:
        seconds_of[test_case.unit_id] = seconds_of.get(test_case.unit_id, 0.0) + test_case.duration
    for test_case in test_cases:
        seconds = seconds_of[test_case.unit_id]
        if seconds > 0:
            test_case.speaking_rate = test_case.ref_words / seconds
