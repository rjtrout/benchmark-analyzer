"""The derived fields: what is worked out here of a test case.

``get_derived`` counts two of them in the reference text, and measures four on the audio:
``read_clip`` reads the audio and ``measure_clip`` measures it, and the measures themselves
are in ``acoustic.py``. The speaking rate is not one test case's
to work out, since a session's chunks share it: ``populate.populate_speaking_rate`` fills it
over a benchmark's whole list.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ba.metadata import acoustic
from ba.paths import Paths

# The derived fields, in the order the pages show them: each one's id, and what it is.
DERIVED = {
    # Counted in the reference text.
    "ref_words": "Words in the reference as written.",
    "alphanumeric": "The reference holds a number, written as digits or as words.",
    # Worked out from the reference's words and the stated length.
    "speaking_rate": "Words a second: the reference's words over the length they were said in, pauses included.",
    # Measured on the audio.
    "level": "Loudness: RMS level in dB below full scale.",
    "snr_est": "Signal-to-noise ratio estimated with WADA-SNR, in dB, from -20 to 100.",
    "bandwidth_est": "The frequency below which nearly all the energy lies, in Hz.",
    "speech_fraction": "The share of the audio that Silero VAD detects as speech.",
}
# The ones measured on the audio: empty until the audio is fetched.
FROM_AUDIO = ("level", "snr_est", "bandwidth_est", "speech_fraction")
# The ones that are true or false. Every other field, given or derived, is a number.
FLAGS = ("alphanumeric",)

# "one" and "oh" are left out: they are far more often not numbers.
_NUMBER_WORDS = {
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion", "percent",
}  # fmt: skip


def _holds_a_number(reference: str) -> bool:
    """Whether the reference as written holds a number: a word with a digit in it, or a number word."""
    for word in re.findall(r"[a-z0-9']+", reference.lower()):
        has_digit = any(character.isdigit() for character in word)
        if has_digit or word in _NUMBER_WORDS:
            return True
    return False


def read_clip(audio: Path) -> tuple[np.ndarray, int]:
    """An audio file as a clip: its samples as one channel, with full scale at 1.0, and its sampling rate."""
    channels, rate = sf.read(str(audio), dtype="float64", always_2d=True)
    samples = channels.mean(axis=1)  # the audio as one channel
    return samples, rate


def measure_clip(samples: np.ndarray, rate: int, paths: Paths) -> dict[str, float]:
    """The four fields measured on the audio, from a clip's samples.

    ``paths`` says where the URGENT authors' code is, which the noise estimate reads.
    """
    return {
        "level": float(acoustic.level_dbfs(samples)),
        "snr_est": acoustic.snr_estimate_db(samples, paths),
        "bandwidth_est": float(acoustic.bandwidth_hz(samples, rate)),
        "speech_fraction": float(acoustic.speech_fraction(samples, rate)),
    }


def get_derived(reference: str, audio: Path | None, paths: Paths) -> dict[str, Any]:
    """What is worked out here of a test case, by field.

    ``reference`` is the test case's reference as written. ``audio`` is its audio file, or
    None while the audio is not fetched: the four fields measured on it are then left out.
    ``paths`` says where the URGENT authors' code is, which the noise estimate reads.
    """
    derived: dict[str, Any] = {
        "ref_words": len(reference.split()),
        "alphanumeric": _holds_a_number(reference),
    }
    if audio is not None:
        samples, rate = read_clip(audio)
        derived.update(measure_clip(samples, rate, paths))
    return derived
