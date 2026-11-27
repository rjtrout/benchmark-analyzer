"""The given fields: what a test case's source states of it.

``get_given`` reads them from the test case's row of its dataset, and from its audio file's
own header. What the source states is taken as it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ba.utils.audio_utils import probe

# The given fields, in the order the pages show them: each one's id, and what it is.
GIVEN = {
    "duration": "Length in seconds, as the source states it.",
    "snr_given": "Signal-to-noise ratio in dB, where the set states it: URGENT's noisy half.",
    "sampling_rate": "The audio file's own sampling rate, known once the audio is fetched.",
}


def get_given(row: dict[str, Any], audio: Path | None) -> dict[str, Any]:
    """What the source states of a test case, by field.

    ``row`` is the test case's row of its dataset. ``audio`` is its audio file, or None
    while the audio is not fetched. A field the source does not state is left out.
    """
    # The leaderboard's sets state a test case's length as audio_length_s, and Earnings22's chunks as duration.
    seconds = row.get("audio_length_s") or row.get("duration") or 0.0
    given: dict[str, Any] = {"duration": float(seconds)}
    if row.get("snr_db") is not None:
        given["snr_given"] = float(row["snr_db"])
    if audio is not None:
        rate, _ = probe(audio)
        given["sampling_rate"] = rate
    return given
