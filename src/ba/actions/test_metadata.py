"""Tests of the metadata fields, on the real project: what ``ba test_metadata`` runs.

They are not part of the test suite. Each sets a field against something else that is known
of the same audio, and gives figures:

    URGENT states the level its noise was mixed at, so the signal-to-noise ratio estimated
    on its noisy half is set against the stated one.
    URGENT's clean half is the same recordings before they were distorted, so the speech
    fraction of the two halves is set against each other.
    A clip changed in a known way says what a field should then read. A sample of URGENT's
    clean clips is changed by us in three ways, one at a time: made louder, for the level;
    given white noise at a known ratio, for the noise estimate; and with its top cut off at
    a known frequency, for the bandwidth.

URGENT's noisy half is more than its clean half with noise. Its authors' code mixes noise
into every recording, reverberates some first, and cuts off the top of the mixture's range
or clips it for some. The level it states is the speech's power over the noise's, each
counted over its non-silent parts alone, before any cutting off or clipping. The authors
have tagged every recording of the test set, by listening, with what was done to it. So
the comparisons of URGENT's halves have a row for each thing done.

Everything here reads. Nothing is kept in the store.
"""

from __future__ import annotations

import csv
import math
from typing import Any

import numpy as np
from scipy.signal import resample_poly
from scipy.stats import spearmanr

from ba.metadata import acoustic
from ba.metadata.derived import read_clip
from ba.paths import Paths
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.utils.audio_utils import audio_file

URGENT_CLEAN = "urgent-clean"  # URGENT's recordings before they were distorted
URGENT_NOISY = "urgent-noisy"  # the same recordings distorted, each stating the level its noise was mixed at

# One recording of URGENT's: the tags its authors gave it, its clean test case and its noisy one.
Recording = tuple[list[str], TestCase, TestCase]

SAMPLE = 100  # how many of URGENT's clean clips are changed by us
SEED = 0  # of the choice of those clips, and of what is done to each
LOUDER_DB = 6.0  # how much louder a clip is made
NOISE_SNRS_DB = (0.0, 20.0)  # white noise is added to a clip at a signal-to-noise ratio from the first to the second
LOWER_RATES_HZ = (8000, 12000, 14000)  # the sampling rates a clip is resampled to and back from, each cutting its top off at half the rate


# ── URGENT's two halves ──────────────────────────────────────────────────────────────────


def _recording(test_case: TestCase) -> str:
    """The recording a test case of URGENT is of: its id without the dataset, ``fileid_945``."""
    return test_case.test_case_id.split("/", 1)[1]


def read_urgent_tags(paths: Paths) -> dict[str, list[str]]:
    """The tags URGENT's authors gave each recording of the test set, under the recording's id.

    They are a file of the authors' analysis code, which is kept as a git submodule. It has a
    line for each recording: its id, a tab, and its tags with semicolons between them.
    """
    tags_of = {}
    with open(paths.urgent_clone / "tagging" / "nonblind_test_tags.tsv", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            tags_of[row["id"]] = row["tags"].split(";")
    return tags_of


def urgent_recordings(store: Store) -> list[Recording]:
    """URGENT's recordings, each as its tags, its clean test case and its noisy one.

    The two halves name a recording alike after the dataset: ``urgent2024_clean/fileid_945``
    and ``urgent2024/fileid_945``, and its tags are under ``fileid_945``. The recordings are
    in the noisy half's order. One whose audio is not measured yet in both halves is left out.
    """
    tags_of = read_urgent_tags(store.paths)
    clean_of = {}
    for clean in store.test_cases.read(URGENT_CLEAN):
        clean_of[_recording(clean)] = clean

    recordings = []
    for noisy in store.test_cases.read(URGENT_NOISY):
        recording = _recording(noisy)
        clean = clean_of[recording]
        if clean.snr_est is None or noisy.snr_est is None:
            continue
        recordings.append((tags_of[recording], clean, noisy))
    if not recordings:
        raise RuntimeError("URGENT's audio is not measured yet. It is fetched and measured by: ba fetch --audio")
    return recordings


def get_urgent_tags(tags: list[str]) -> str:
    """What URGENT did to a recording beyond adding noise, as its tags tell it: ``reverberated, clipped``.

    The tags have three strengths of reverberation, which are one thing here, and
    ``bandwidth_limitation`` is the top of the mixture's range cut off. A recording that is
    reverberated is reverberated first, and one is cut off or clipped and never both. One
    with none of these tags is ``noise only``.
    """
    done = []
    if "weak_reverberation" in tags or "medium_reverberation" in tags or "strong_reverberation" in tags:
        done.append("reverberated")
    if "bandwidth_limitation" in tags:
        done.append("cut off")
    if "clipping" in tags:
        done.append("clipped")
    if not done:
        return "noise only"
    return ", ".join(done)


def _row(name: str, expected: np.ndarray, measured: np.ndarray) -> dict[str, Any]:
    """One row of a comparison: its name, how many recordings, the mean of each reading, and how the two differ and agree.

    ``difference`` is the mean of the measured reading less the expected one. ``apart`` is
    how far apart the two are on average, whichever is the higher. ``rank_correlation`` says
    whether the two put the recordings in the same order. It is None where every recording
    has the one expected reading, which leaves nothing to rank them by.
    """
    rank_correlation = None
    if len(np.unique(expected)) > 1:
        rank_correlation = float(spearmanr(expected, measured).statistic)
    return {
        "name": name,
        "test_cases": len(expected),
        "expected": float(expected.mean()),
        "measured": float(measured.mean()),
        "difference": float((measured - expected).mean()),
        "apart": float(np.abs(measured - expected).mean()),
        "rank_correlation": rank_correlation,
    }


def _compared(done: list[str], expected: list[float], measured: list[float]) -> list[dict[str, Any]]:
    """Two readings of URGENT's recordings set against each other, by what was done to each recording.

    There is a row for each thing done, in the alphabet's order, and a last one for every
    recording: see ``_row``.
    """
    done_values = np.array(done)
    expected_values = np.array(expected)
    measured_values = np.array(measured)

    rows = []
    for thing_done in np.unique(done_values):  # in the alphabet's order
        had_it_done = done_values == thing_done
        rows.append(_row(str(thing_done), expected_values[had_it_done], measured_values[had_it_done]))
    rows.append(_row("all", expected_values, measured_values))
    return rows


def snr_against_stated(recordings: list[Recording]) -> list[dict[str, Any]]:
    """The signal-to-noise ratio estimated on URGENT's noisy test cases against the one the set states.

    The stated ratio is the expected reading, and the estimate the measured one: see ``_compared``.
    """
    done = [get_urgent_tags(tags) for tags, _, _ in recordings]
    stated = [noisy.snr_given for _, _, noisy in recordings]
    estimated = [noisy.snr_est for _, _, noisy in recordings]
    return _compared(done, expected=stated, measured=estimated)


def clean_against_noisy(recordings: list[Recording], field: str) -> list[dict[str, Any]]:
    """A field as measured on URGENT's clean test cases against the same recordings in its noisy half.

    The speech is the same in both halves. The clean half's reading is the expected one,
    and the noisy half's the measured one: see ``_compared``.
    """
    done = [get_urgent_tags(tags) for tags, _, _ in recordings]
    clean_values = [getattr(clean, field) for _, clean, _ in recordings]
    noisy_values = [getattr(noisy, field) for _, _, noisy in recordings]
    return _compared(done, expected=clean_values, measured=noisy_values)


# ── experiments of our own on URGENT's clean clips ───────────────────────────────────────


def _sample(store: Store, rng: np.random.Generator) -> list[TestCase]:
    """The clean clips of URGENT's that are changed by us: ``SAMPLE`` of them chosen by ``rng``, in the list's order."""
    test_cases = store.test_cases.read(URGENT_CLEAN)
    sample_size = min(SAMPLE, len(test_cases))
    chosen = sorted(rng.choice(len(test_cases), size=sample_size, replace=False))
    return [test_cases[index] for index in chosen]


def _read(store: Store, test_case: TestCase) -> tuple[np.ndarray, int]:
    """A clean clip's audio: its samples and its sampling rate. Audio that is not on disk is an error that says what fetches it."""
    audio = audio_file(store.paths, test_case)
    if audio is None:
        raise RuntimeError(f"The audio of {URGENT_CLEAN} is not on disk. It is fetched by: ba fetch --audio")
    return read_clip(audio)


def louder(samples: np.ndarray, decibels: float) -> np.ndarray:
    """A clip made louder by so many decibels. It is not cut off at full scale."""
    return samples * 10.0 ** (decibels / 20.0)


def level_against_gain(store: Store) -> dict[str, Any]:
    """The level of a sample of URGENT's clean clips made louder, against their level before and the gain.

    Each clip is made ``LOUDER_DB`` louder. Its level before and that much more is the
    expected reading, and the level of the louder clip the measured one. The answer is one
    row: see ``_row``.
    """
    rng = np.random.default_rng(SEED)
    expected = []
    measured = []
    for test_case in _sample(store, rng):
        samples, _ = _read(store, test_case)
        expected.append(acoustic.level_dbfs(samples) + LOUDER_DB)
        measured.append(acoustic.level_dbfs(louder(samples, LOUDER_DB)))
    return _row(f"{LOUDER_DB:g} dB louder by us", np.array(expected), np.array(measured))


def with_white_noise(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """A clip with white noise added to it, at a signal-to-noise ratio of ``snr_db``.

    The ratio is of the clip's power to the noise's, each over the clip's whole length,
    pauses included.
    """
    clip_power = float(np.mean(samples**2))
    noise_power = clip_power / 10.0 ** (snr_db / 10.0)
    noise = rng.normal(0.0, math.sqrt(noise_power), size=len(samples))
    return samples + noise


def snr_against_noise(store: Store) -> dict[str, Any]:
    """The signal-to-noise ratio estimated on a sample of URGENT's clean clips given white noise, against the ratio it was added at.

    Each clip is given white noise at a ratio of its own, drawn from ``NOISE_SNRS_DB``. That
    ratio is the expected reading, and the estimate on the clip with the noise the measured
    one. The answer is one row: see ``_row``.
    """
    rng = np.random.default_rng(SEED)
    lowest, highest = NOISE_SNRS_DB
    expected = []
    measured = []
    for test_case in _sample(store, rng):
        samples, _ = _read(store, test_case)
        snr_db = float(rng.uniform(lowest, highest))
        expected.append(snr_db)
        measured.append(acoustic.snr_estimate_db(with_white_noise(samples, snr_db, rng), store.paths))
    return _row("white noise by us", np.array(expected), np.array(measured))


def resample_down_and_up(samples: np.ndarray, rate: int, lower_rate: int) -> np.ndarray:
    """A clip passed through a lower sampling rate and back, at its own length.

    The lower rate holds nothing above half of itself, so that is where the clip's top is
    cut off. It is how URGENT's code cuts the top off a mixture.
    """
    shared = math.gcd(rate, lower_rate)
    down = resample_poly(samples, lower_rate // shared, rate // shared)
    back_up = resample_poly(down, rate // shared, lower_rate // shared)
    return back_up[: len(samples)]


def bandwidth_against_cut(store: Store) -> list[dict[str, Any]]:
    """The bandwidth of a sample of URGENT's clean clips resampled to a lower rate and back, against the cut that makes.

    There is a row for each rate of ``LOWER_RATES_HZ``, in that order: see ``_row``. A pass
    through a rate cuts off what is above half of it, so that half is the expected reading,
    and the bandwidth of the resampled clip is the measured one. A clip that read under the
    cut before is left out of that row: the pass takes nothing from it.
    """
    rng = np.random.default_rng(SEED)
    sample = _sample(store, rng)
    rows = []
    for lower_rate in LOWER_RATES_HZ:
        cut_at = lower_rate / 2.0
        measured = []
        for test_case in sample:
            samples, rate = _read(store, test_case)
            if acoustic.bandwidth_hz(samples, rate) <= cut_at:
                continue
            measured.append(acoustic.bandwidth_hz(resample_down_and_up(samples, rate, lower_rate), rate))
        expected = np.full(len(measured), cut_at)
        rows.append(_row(f"resampled to {lower_rate} Hz", expected, np.array(measured)))
    return rows
