"""``ba test_metadata``: tests of the metadata fields, on the real project. Not part of the test suite."""

from __future__ import annotations

from typing import Any

import typer

from ba.actions.test_metadata import (
    LOUDER_DB,
    NOISE_SNRS_DB,
    URGENT_CLEAN,
    URGENT_NOISY,
    bandwidth_against_cut,
    clean_against_noisy,
    level_against_gain,
    snr_against_noise,
    snr_against_stated,
    urgent_recordings,
)
from ba.cli.common import guard, setup
from ba.utils.display_utils import fill, plural, table


def _signed(value: float, decimals: int) -> str:
    """A figure with its sign, to so many decimals. One too small to show is a plain zero, with no sign."""
    if round(value, decimals) == 0:
        return f"{0:.{decimals}f}"
    return f"{value:+.{decimals}f}"


def _comparison(compared: list[dict[str, Any]], expected: str, measured: str, decimals: int, rank: bool = True) -> str:
    """One comparison as a table: a row for each thing done, under what the two readings are called.

    ``rank`` is whether the table has the rank correlation of the two readings. It is left
    out where every recording of a row has the one expected reading.
    """
    headers = ["what was done", "test cases", expected, measured, "difference", "apart"]
    figures = [1, 2, 3, 4, 5]  # the columns that hold figures
    if rank:
        headers.append("rank")
        figures.append(6)

    rows = []
    for row in compared:
        cells = [
            row["name"],
            row["test_cases"],
            f"{row['expected']:.{decimals}f}",
            f"{row['measured']:.{decimals}f}",
            _signed(row["difference"], decimals),
            f"{row['apart']:.{decimals}f}",
        ]
        if rank:
            cells.append(f"{row['rank_correlation']:.2f}")
        rows.append(cells)
    return table(headers, rows, right=figures)


def _echo_test(heading: str, note: str, comparison: str) -> None:
    """Print one test, set apart from the one before it: its heading, a note on it if it has one, and its table."""
    typer.echo("\n\n" + heading)
    if note:
        typer.echo("\n".join(fill(note)))
    typer.echo("\n" + comparison)


def test_metadata() -> None:
    """Test the metadata fields against what else is known of the same audio.

    Not part of the test suite: it reads the real project and its audio, gives figures, and
    saves nothing. URGENT states the level its noise was mixed at, and its clean half is the
    same recordings before they were distorted. So the estimated SNR is set against the
    stated one, and the speech fraction of the two halves against each other. The noisy
    half is noise and, for some recordings, reverberation, a cut-off top or clipping.
    URGENT's authors tagged which, so those tables have a row for each thing done. Then
    come three experiments of our own on some of URGENT's clean clips, each changed in one
    known way: made louder for the level, given white noise for the SNR estimate, and cut
    off at the top for the bandwidth.
    """
    with guard():
        _, _, store = setup()

        about_urgent = "The first two tables set URGENT's two halves against each other. Every noisy recording has noise added. Some were reverberated first, and some then had their top cut off or were clipped. URGENT's authors tagged which, so each table has a row for each thing done."
        note = "In every table, difference is the mean of the second reading less the first, apart is how far apart the two are on average, whichever is the higher, and rank is the rank correlation of the two: 1.00 when they rise and fall together."
        typer.echo("\n".join(fill(about_urgent)))
        typer.echo("\n".join(fill(note)))

        recordings = urgent_recordings(store)

        # URGENT's noisy half: the estimate against what the set states.
        heading = f"SNR estimate against the SNR that URGENT states, in dB ({URGENT_NOISY})"
        _echo_test(heading, "", _comparison(snr_against_stated(recordings), "stated", "estimated", decimals=2))

        # URGENT's two halves: the same speech, before it was distorted and after.
        heading = "Speech fraction of URGENT's clean test cases against the same in its noisy half"
        compared = clean_against_noisy(recordings, "speech_fraction")
        _echo_test(heading, "", _comparison(compared, "clean", "noisy", decimals=3))

        # Experiments of our own: a sample of URGENT's clean clips, each changed in one known way.
        made_louder = level_against_gain(store)
        clips = plural(made_louder["test_cases"], "clip")
        about_ours = f"The last three tables are experiments of our own on {clips} of {URGENT_CLEAN}. In each, every clip is changed in one known way, in memory, and the field that should follow the change is measured on it."
        typer.echo("\n\n" + "\n".join(fill(about_ours)))

        heading = f"Level of the clips made {LOUDER_DB:g} dB louder, against their level before plus {LOUDER_DB:g}, in dB"
        _echo_test(heading, "", _comparison([made_louder], "expected", "measured", decimals=2))

        lowest, highest = NOISE_SNRS_DB
        heading = "SNR estimate of the clips given white noise, against the SNR added, in dB"
        note = f"Each clip is given white noise at a signal-to-noise ratio of its own, from {lowest:g} to {highest:g} dB."
        _echo_test(heading, note, _comparison([snr_against_noise(store)], "added", "estimated", decimals=2))

        heading = "Bandwidth of clips resampled to a lower rate and back, against the cut, in Hz"
        note = "A sampling rate holds nothing above half of itself. So a clip resampled down to a rate and back up has its top cut off at half that rate, which is what its bandwidth should then read. A clip that read under the cut before is left out: the resampling takes nothing from it."
        _echo_test(heading, note, _comparison(bandwidth_against_cut(store), "cut at", "estimated", decimals=0, rank=False))
