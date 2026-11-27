from __future__ import annotations

import math

import numpy as np
import pytest

from ba.metadata import acoustic
from ba.paths import Paths

RATE = 16000


def lowpass(samples: np.ndarray, cutoff_hz: float, rate: int = RATE) -> np.ndarray:
    """Remove everything above the cutoff, as a pass through a lower sampling rate does."""
    spectrum = np.fft.rfft(samples)
    spectrum[np.fft.rfftfreq(len(samples), 1 / rate) > cutoff_hz] = 0
    return np.fft.irfft(spectrum, len(samples))


def sine(freq: float, amplitude: float, seconds: float = 1.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def speechlike() -> np.ndarray:
    """Harmonic bursts with varying length, pitch and level, separated by pauses."""
    rng = np.random.default_rng(0)
    step = int(0.02 * RATE)
    total = 8 * RATE
    t = np.arange(total) / RATE
    samples = np.zeros(total)
    start = step * int(rng.integers(5, 15))
    while start < total:
        frames = int(rng.integers(10, 30))
        end = min(total, start + step * frames)
        pitch = rng.uniform(110, 220)
        harmonics = sum(np.sin(2 * np.pi * pitch * k * t[start:end]) / k for k in range(1, 16))
        samples[start:end] = rng.uniform(0.5, 1.0) * harmonics
        start = end + step * int(rng.integers(5, 20))
    return 0.25 * samples / np.abs(samples).max()


def band_limited_noise(seconds: float = 4.0, seed: int = 3) -> np.ndarray:
    """White noise at 16 kHz that has been down to 8 kHz and back."""
    noise = 0.1 * np.random.default_rng(seed).standard_normal(int(seconds * RATE))
    return lowpass(noise, 4000)


# ── level ────────────────────────────────────────────────────────────────────────────────


def test_level_of_a_sine_is_its_rms_in_db_below_full_scale():
    assert acoustic.level_dbfs(sine(440, 1.0)) == pytest.approx(-3.01, abs=0.01)
    assert acoustic.level_dbfs(sine(440, 0.5)) == pytest.approx(-9.03, abs=0.01)
    assert acoustic.level_dbfs(sine(440, 0.05)) == pytest.approx(-29.03, abs=0.01)


def test_silence_has_a_floor_instead_of_minus_infinity():
    assert acoustic.level_dbfs(np.zeros(RATE)) == -120.0
    assert acoustic.level_dbfs(np.zeros(0)) == -120.0
    assert acoustic.level_dbfs(np.full(RATE, 1e-9)) == -120.0


# ── speech activity ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("rate", "vad_rate"), [(8000, 8000), (16000, 16000), (44100, 16000)])
def test_active_mask_maps_silero_intervals_to_the_original_samples(monkeypatch, rate, vad_rate):
    samples = np.full(rate - 1, 0.1)

    def detect(audio, sampling_rate):
        assert sampling_rate == vad_rate
        assert audio.dtype == np.float32
        assert audio.shape == (math.ceil(len(samples) * vad_rate / rate),)
        return [
            {"start": vad_rate // 4, "end": vad_rate // 2},
            {"start": 3 * vad_rate // 4, "end": len(audio)},
        ]

    monkeypatch.setattr(acoustic, "_silero_detector", lambda: detect)
    expected = np.zeros(samples.shape, dtype=bool)
    expected[rate // 4 : rate // 2] = True
    expected[3 * rate // 4 :] = True

    mask = acoustic.active_mask(samples, rate)
    assert mask.dtype == bool
    assert np.array_equal(mask, expected)
    assert acoustic.speech_fraction(samples, rate) == pytest.approx(expected.mean())


def test_silero_can_return_no_speech_for_a_short_clip(monkeypatch):
    def detect(audio, sampling_rate):
        assert audio.shape == (80,)
        assert sampling_rate == RATE
        return []

    monkeypatch.setattr(acoustic, "_silero_detector", lambda: detect)
    samples = sine(440, 0.3, seconds=0.005)
    assert np.array_equal(acoustic.active_mask(samples, RATE), np.zeros(len(samples), dtype=bool))
    assert acoustic.speech_fraction(samples, RATE) == 0.0


def test_silence_and_empty_input_skip_silero(monkeypatch):
    def load_detector():
        pytest.fail("Silero should not be loaded for silence")

    monkeypatch.setattr(acoustic, "_silero_detector", load_detector)
    for samples in (np.zeros(RATE), np.zeros(0)):
        assert np.array_equal(acoustic.active_mask(samples, RATE), np.zeros(len(samples), dtype=bool))
        assert acoustic.speech_fraction(samples, RATE) == 0.0


# ── signal-to-noise ratio ────────────────────────────────────────────────────────────────


def test_snr_estimate_has_a_floor_for_silence_and_empty_input(paths):
    assert acoustic.snr_estimate_db(np.zeros(RATE), paths) == -20.0
    assert acoustic.snr_estimate_db(np.zeros(0), paths) == -20.0


def test_wada_snr_interpolates_the_table_independently_of_gain(paths):
    # Two magnitudes, 1 and 1/16, give log(17/8), between the 12 and 13 dB entries.
    samples = np.array([-1.0, 1.0, -0.0625, 0.0625])
    expected = pytest.approx(12.987, abs=0.002)
    assert acoustic.snr_estimate_db(samples, paths) == expected
    assert acoustic.snr_estimate_db(samples * 0.5, paths) == expected


def test_the_estimate_runs_from_the_floor_to_the_top_of_wadas_table(paths):
    # Below the table there is no speech the method can tell from the noise: the floor, as for silence.
    # The script itself answers 0.41 there, the table's first coefficient.
    assert acoustic.snr_estimate_db(np.array([-1.0, 1.0]), paths) == -20.0
    noise = 0.1 * np.random.default_rng(0).standard_normal(RATE)
    assert acoustic.snr_estimate_db(noise, paths) == -20.0
    assert acoustic.snr_estimate_db(sine(440, 0.3), paths) == -20.0
    # Above the table the clip is cleaner than the method can tell: the table's last level.
    assert acoustic.snr_estimate_db(np.array([-1.0, 1.0, -0.001, 0.001]), paths) == pytest.approx(100.0)


def test_the_wada_script_is_found_through_the_projects_paths(tmp_path, paths):
    samples = np.array([-1.0, 1.0, -0.0625, 0.0625])
    assert acoustic.snr_estimate_db(samples, paths) == pytest.approx(12.987, abs=0.002)
    # A project whose clone is not there says where it looked and what to run.
    elsewhere = Paths(root=tmp_path / "elsewhere")
    with pytest.raises(RuntimeError, match="git submodule update --init") as error:
        acoustic.snr_estimate_db(samples, elsewhere)
    assert str(elsewhere.urgent_clone) in str(error.value)


# ── bandwidth ────────────────────────────────────────────────────────────────────────────


def test_bandwidth_of_audio_that_has_been_through_8_khz_is_at_most_4_khz():
    bandwidth = acoustic.bandwidth_hz(band_limited_noise(), RATE)
    assert 3000 < bandwidth <= 4050


def test_bandwidth_of_full_band_noise_is_near_half_the_rate():
    noise = 0.1 * np.random.default_rng(5).standard_normal(4 * RATE)
    assert acoustic.bandwidth_hz(noise, RATE) == pytest.approx(8000, abs=300)
    assert acoustic.bandwidth_hz(noise, RATE) <= 8000


def test_bandwidth_tells_wide_band_speech_from_narrow_though_little_energy_is_high():
    # Speech keeps well under a hundredth of its energy above 4 kHz. Here it is a 300th,
    # as hiss between 4 and 7 kHz, so a line drawn at 99% of the energy would miss it.
    samples = speechlike()
    spectrum = np.fft.rfft(np.random.default_rng(8).standard_normal(len(samples)))
    freqs = np.fft.rfftfreq(len(samples), 1 / RATE)
    hiss = np.fft.irfft(np.where((freqs > 4000) & (freqs < 7000), spectrum, 0), len(samples))
    wide = samples + hiss * np.sqrt(np.mean(samples**2) / np.mean(hiss**2) / 300)
    narrow = lowpass(wide, 4000)
    assert acoustic.bandwidth_hz(wide, RATE) > 5000
    assert acoustic.bandwidth_hz(narrow, RATE) <= 4000
    assert acoustic.bandwidth_hz(wide, RATE, fraction=0.99) < 4000


def test_bandwidth_of_a_tone_is_its_frequency():
    assert acoustic.bandwidth_hz(sine(1000, 0.3, seconds=2.0), RATE) == pytest.approx(1000, abs=100)


def test_bandwidth_of_silence_is_zero():
    assert acoustic.bandwidth_hz(np.zeros(RATE), RATE) == 0.0
    assert acoustic.bandwidth_hz(np.zeros(0), RATE) == 0.0
