"""Acoustic estimators: level, speech activity, signal-to-noise ratio and bandwidth.

Each takes one clip, which is a test case's audio, as mono float samples with full scale at
1.0. Speech activity is detected with Silero VAD, and the signal-to-noise ratio is estimated
with the URGENT authors' WADA-SNR. Each docstring says what the number means.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from threading import Lock
from types import ModuleType

import numpy as np
from scipy.signal import resample_poly, welch

from ba.paths import Paths

FLOOR_DBFS = -120.0  # what silence reads as, in place of minus infinity
SNR_FLOOR_DB = -20.0  # the lowest signal-to-noise ratio reported: the bottom of WADA-SNR's table

_SILERO_LOCK = Lock()


def _db(power: float) -> float:
    """A power as decibels, and never below the floor: silence has no logarithm."""
    if power <= 0:
        return FLOOR_DBFS
    return max(FLOOR_DBFS, 10.0 * float(np.log10(power)))


def level_dbfs(samples: np.ndarray) -> float:
    """RMS level in dB relative to full scale.

    A full-scale sine reads -3.01. Silence reads -120.0, not minus infinity.
    """
    if samples.size == 0:
        return FLOOR_DBFS
    mean_power = float(np.mean(samples**2))
    return _db(mean_power)


@cache
def _silero_detector() -> Callable[[np.ndarray, int], list[dict[str, int]]]:
    import torch

    # Silero sets PyTorch's thread count at import. Preserve it for other models.
    threads = torch.get_num_threads()
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    finally:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)

    model = load_silero_vad()

    def detect(audio: np.ndarray, rate: int) -> list[dict[str, int]]:
        return get_speech_timestamps(
            torch.from_numpy(audio), model, sampling_rate=rate, speech_pad_ms=0
        )

    return detect


def active_mask(samples: np.ndarray, rate: int) -> np.ndarray:
    """Samples in Silero's speech intervals, as a boolean mask at the original sample rate.

    Silero accepts 8 kHz and 16 kHz audio. Other rates are resampled to 16 kHz for detection.
    The model uses its default minimum speech duration of 250 ms with no boundary padding.
    """
    mask = np.zeros(len(samples), dtype=bool)
    if not np.any(samples):
        return mask

    audio = np.ascontiguousarray(samples, dtype=np.float32)
    vad_rate = rate if rate in (8000, 16000) else 16000
    if rate != vad_rate:
        audio = resample_poly(audio, vad_rate, rate)

    # Silero resets and updates recurrent state for each clip.
    with _SILERO_LOCK:
        segments = _silero_detector()(audio, vad_rate)

    for segment in segments:
        start = round(segment["start"] * rate / vad_rate)
        end = round(segment["end"] * rate / vad_rate)
        mask[start:end] = True
    return mask


def speech_fraction(samples: np.ndarray, rate: int) -> float:
    """The share of the clip that ``active_mask`` calls speech, from 0 to 1."""
    mask = active_mask(samples, rate)
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


@cache
def _wada_script(clone: Path) -> ModuleType:
    """The URGENT authors' WADA-SNR script, read from their code, which is kept as a git submodule."""
    script = clone / "wada_snr" / "calculate_wada_snr.py"
    if not script.is_file():
        raise RuntimeError(f"The URGENT authors' code is not at {clone}. Run: git submodule update --init")
    spec = spec_from_file_location("urgent_wada_snr", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snr_estimate_db(samples: np.ndarray, paths: Paths) -> float:
    """Signal-to-noise ratio in dB, estimated from the clip's amplitudes with WADA-SNR.

    The estimator is the URGENT authors' script, at ``paths.urgent_clone``. It takes clean
    speech to have a gamma distribution of amplitudes and the noise added to it a Gaussian
    one, and reads the ratio off a table that runs from -20 to 100 dB. So the answer runs
    from -20 to 100 as well. A clip cleaner than the table can tell reads 100. A clip with
    no speech that the method can tell from the noise reads -20, as an empty or silent one
    does.
    """
    if not np.any(samples):
        return SNR_FLOOR_DB
    script = _wada_script(paths.urgent_clone)
    estimate = script.wada_snr(samples)
    if estimate is None or not np.isfinite(estimate):
        return SNR_FLOOR_DB
    # Below its table the script answers with the table's first coefficient, 0.41, where the
    # table's lowest level is meant.
    if np.isclose(estimate, script.Gvals[0]):
        return SNR_FLOOR_DB
    return float(estimate)


def bandwidth_hz(samples: np.ndarray, rate: int, fraction: float = 0.999) -> float:
    """The frequency in Hz below which nearly all of the clip's energy lies.

    Read from a spectrum averaged over overlapping 32 ms windows. Audio that has been
    through 8 kHz reads at most 4000, and full-band noise reads half the rate.

    "Nearly all" is 99.9%, not 99%. Speech keeps a hundredth or less of its energy above
    4 kHz, so a line at 99% reads wide-band speech and telephone speech alike.
    """
    if not np.any(samples):
        return 0.0
    size = 2 ** int(round(np.log2(0.032 * rate)))
    if len(samples) < size:
        samples = np.pad(samples, (0, size - len(samples)))
    _, spectrum = welch(
        np.asarray(samples, dtype=np.float64),
        fs=rate,
        window=np.hanning(size),
        noverlap=size // 2,
        detrend=False,
    )
    # Undo Welch's doubling of the interior bins to retain the previous FFT weighting.
    spectrum[1:-1] *= 0.5
    cumulative = np.cumsum(spectrum)
    target = fraction * cumulative[-1]
    index = int(np.searchsorted(cumulative, target))
    below = 0.0  # the energy in the bins under this one
    if index > 0:
        below = cumulative[index - 1]
    # Bin k stands for the band half a bin either side of its frequency.
    position = index - 0.5 + (target - below) / spectrum[index]
    frequency = position * rate / size
    highest_possible = rate / 2.0
    return float(min(max(frequency, 0.0), highest_possible))
