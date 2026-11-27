"""`ba test_metadata` and its action, on a small made-up project.

The command itself is run on the real project, and is no part of this suite. These are the
tests of its code: how it pairs URGENT's halves, how it tells what was done to a recording,
what it sets against what, and the experiments it makes on URGENT's clean clips.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from ba.actions import test_metadata as checks
from ba.cli import common, main
from ba.metadata import acoustic
from ba.paths import Paths
from ba.store.store import Store
from ba.store.test_cases import TestCase

RATE = 16000


def with_tags(paths: Paths, tags_of: dict[str, str]) -> Paths:
    """The scratch project with a stand-in for URGENT's clone: these tags, and the real clone's noise estimator.

    The real clone is the repo's, and is only read. The stand-in is under the scratch project.
    """
    stand_in = paths.root / "urgent"
    (stand_in / "tagging").mkdir(parents=True)
    (stand_in / "wada_snr").symlink_to(paths.urgent_clone / "wada_snr")
    lines = ["id\ttags"]
    for recording, tags in tags_of.items():
        lines.append(f"{recording}\t{tags}")
    (stand_in / "tagging" / "nonblind_test_tags.tsv").write_text("\n".join(lines) + "\n")
    return Paths(root=paths.root, configs=paths.configs, clone=paths.clone, urgent_clone=stand_in)


def save_urgent(paths: Paths, recordings: list[tuple[str, dict, dict]]) -> Store:
    """URGENT's two halves in a scratch store: each recording's tags, its clean fields, and its fields with noise."""
    tags_of = {}
    clean, noisy = [], []
    for number, (tags, clean_fields, noisy_fields) in enumerate(recordings):
        tags_of[f"fileid_{number}"] = tags
        clean.append(TestCase(f"urgent2024_clean/fileid_{number}", **clean_fields))
        noisy.append(TestCase(f"urgent2024/fileid_{number}", **noisy_fields))
    store = Store(with_tags(paths, tags_of))
    store.test_cases.write("urgent-clean", clean)
    store.test_cases.write("urgent-noisy", noisy)
    return store


def spiky_clip(seed: int, seconds: float = 1.0) -> np.ndarray:
    """A clip of rare high peaks over a low floor, as speech is, and not as a tone or steady noise is."""
    peaks = np.random.default_rng(seed).standard_normal(int(seconds * RATE)) ** 3
    return 0.5 * peaks / np.abs(peaks).max()


def save_clean_clips(paths: Paths, clips: list[np.ndarray]) -> Store:
    """URGENT's clean half with its audio on disk: these clips, in this order."""
    store = Store(paths)
    folder = paths.audio / "urgent2024_clean"
    folder.mkdir(parents=True)
    test_cases = []
    for number, clip in enumerate(clips):
        audio_path = f"urgent2024_clean/fileid_{number}.flac"
        sf.write(str(paths.audio / audio_path), clip, RATE)
        test_cases.append(TestCase(f"urgent2024_clean/fileid_{number}", audio_path=audio_path, duration=1.0))
    store.test_cases.write("urgent-clean", test_cases)
    return store


# ── URGENT's two halves ──────────────────────────────────────────────────────────────────


def test_urgents_tags_are_read_for_each_recording(paths):
    paths = with_tags(paths, {"fileid_1": "male;simulated;clipping;background_noise", "fileid_2": "simulated;background_noise"})
    assert checks.read_urgent_tags(paths) == {
        "fileid_1": ["male", "simulated", "clipping", "background_noise"],
        "fileid_2": ["simulated", "background_noise"],
    }


def test_what_was_done_to_a_recording_is_told_from_its_tags():
    assert checks.get_urgent_tags(["male", "simulated", "background_noise"]) == "noise only"
    assert checks.get_urgent_tags(["simulated", "bandwidth_limitation"]) == "cut off"
    assert checks.get_urgent_tags(["clipping", "simulated"]) == "clipped"
    # URGENT's three strengths of reverberation are one thing here.
    for strength in ("weak", "medium", "strong"):
        assert checks.get_urgent_tags(["simulated", f"{strength}_reverberation"]) == "reverberated"
    # A recording is reverberated first, and then cut off or clipped.
    assert checks.get_urgent_tags(["bandwidth_limitation", "medium_reverberation"]) == "reverberated, cut off"
    assert checks.get_urgent_tags(["clipping", "strong_reverberation"]) == "reverberated, clipped"


def test_urgents_two_halves_are_paired_by_recording_each_with_its_tags(paths):
    store = Store(with_tags(paths, {"fileid_1": "simulated;clipping", "fileid_2": "simulated"}))
    store.test_cases.write("urgent-clean", [TestCase("urgent2024_clean/fileid_2", snr_est=40.0), TestCase("urgent2024_clean/fileid_1", snr_est=50.0)])
    store.test_cases.write("urgent-noisy", [TestCase("urgent2024/fileid_1", snr_est=5.0), TestCase("urgent2024/fileid_2", snr_est=6.0)])
    recordings = checks.urgent_recordings(store)
    # In the noisy half's order, each with its tags and the clean test case of the same recording.
    assert [(tags, clean.test_case_id, noisy.test_case_id) for tags, clean, noisy in recordings] == [
        (["simulated", "clipping"], "urgent2024_clean/fileid_1", "urgent2024/fileid_1"),
        (["simulated"], "urgent2024_clean/fileid_2", "urgent2024/fileid_2"),
    ]


def test_a_recording_that_is_not_measured_in_both_halves_is_left_out(paths):
    measured = {"snr_est": 40.0}
    # The first is measured in both halves, the second in the noisy half alone, the third in the clean half alone.
    store = save_urgent(paths, [("simulated", measured, {"snr_est": 5.0}), ("simulated", {}, {"snr_est": 6.0}), ("simulated", measured, {})])
    assert [noisy.test_case_id for _, _, noisy in checks.urgent_recordings(store)] == ["urgent2024/fileid_0"]


def test_no_recording_measured_says_what_to_run(paths):
    store = save_urgent(paths, [("simulated", {}, {}), ("simulated", {}, {})])
    with pytest.raises(RuntimeError, match="ba fetch --audio"):
        checks.urgent_recordings(store)


def test_the_snr_estimate_is_set_against_the_stated_one_by_what_was_done():
    clean = TestCase("urgent2024_clean/x")  # the clean half plays no part here
    recordings = []
    for tags, stated, estimated in (
        (["simulated"], 2.0, -1.0),
        (["simulated", "clipping"], 0.0, -10.0),
        (["simulated"], 6.0, 9.0),
        (["simulated", "clipping"], 10.0, 2.0),
        (["simulated"], 4.0, 4.0),
        (["simulated", "clipping"], 20.0, -4.0),
    ):
        recordings.append((tags, clean, TestCase("urgent2024/x", snr_given=stated, snr_est=estimated)))
    # A row for each thing done, in the alphabet's order, and one for every recording.
    assert checks.snr_against_stated(recordings) == [
        # Clipped: 10, 8 and 24 dB low, and the one stated highest reads second.
        {"name": "clipped", "test_cases": 3, "expected": 10.0, "measured": -4.0, "difference": -14.0, "apart": 14.0, "rank_correlation": pytest.approx(0.5)},
        # Noise only: 3 dB low, 3 dB high and on the mark, in the order of the stated ones.
        {"name": "noise only", "test_cases": 3, "expected": 4.0, "measured": 4.0, "difference": 0.0, "apart": 2.0, "rank_correlation": pytest.approx(1.0)},
        {"name": "all", "test_cases": 6, "expected": 7.0, "measured": 0.0, "difference": -7.0, "apart": 8.0, "rank_correlation": pytest.approx(1 - 6 * 26 / 210)},
    ]


def test_a_field_of_the_clean_half_is_set_against_the_same_recordings_with_noise():
    recordings = []
    for tags, clean_value, noisy_value in (
        (["simulated"], 0.5, 0.375),
        (["simulated", "weak_reverberation"], 0.8125, 0.25),
        (["simulated"], 0.625, 0.5),
        (["simulated", "medium_reverberation"], 0.875, 0.1875),
        (["simulated"], 0.75, 0.625),
        (["simulated", "strong_reverberation"], 0.9375, 0.3125),
    ):
        recordings.append((tags, TestCase("urgent2024_clean/x", speech_fraction=clean_value), TestCase("urgent2024/x", speech_fraction=noisy_value)))
    # The clean half's reading is the expected one.
    assert checks.clean_against_noisy(recordings, "speech_fraction") == [
        {"name": "noise only", "test_cases": 3, "expected": 0.625, "measured": 0.5, "difference": -0.125, "apart": 0.125, "rank_correlation": pytest.approx(1.0)},
        {"name": "reverberated", "test_cases": 3, "expected": 0.875, "measured": 0.25, "difference": -0.625, "apart": 0.625, "rank_correlation": pytest.approx(0.5)},
        # Over all six the reverberated ones, which read highest clean, read lowest with noise.
        {"name": "all", "test_cases": 6, "expected": 0.75, "measured": 0.375, "difference": -0.375, "apart": 0.375, "rank_correlation": pytest.approx(-0.6)},
    ]


# ── experiments of our own on URGENT's clean clips ───────────────────────────────────────


def test_louder_raises_a_clips_level_by_the_decibels_given():
    clip = spiky_clip(seed=1)
    assert acoustic.level_dbfs(checks.louder(clip, 6.0)) - acoustic.level_dbfs(clip) == pytest.approx(6.0)


def test_white_noise_is_added_at_the_ratio_asked_for_and_the_same_from_the_same_seed():
    clip = spiky_clip(seed=2, seconds=4.0)
    with_noise = checks.with_white_noise(clip, 10.0, np.random.default_rng(0))
    noise = with_noise - clip
    assert 10 * np.log10(np.mean(clip**2) / np.mean(noise**2)) == pytest.approx(10.0, abs=0.2)
    assert np.array_equal(with_noise, checks.with_white_noise(clip, 10.0, np.random.default_rng(0)))


def test_the_level_of_a_louder_clip_is_set_against_its_level_before_and_the_gain(paths, monkeypatch):
    monkeypatch.setattr(checks, "SAMPLE", 3)
    monkeypatch.setattr(checks, "LOUDER_DB", 6.0)
    store = save_clean_clips(paths, [spiky_clip(seed=number) for number in range(4)])
    row = checks.level_against_gain(store)
    assert (row["name"], row["test_cases"]) == ("6 dB louder by us", 3)  # a sample of the four
    # What each louder clip should read is its level before and 6 dB more: here of three different clips of the four.
    levels = [acoustic.level_dbfs(spiky_clip(seed=number)) for number in range(4)]
    of_three = [np.mean(levels[:left_out] + levels[left_out + 1 :]) + 6.0 for left_out in range(4)]
    assert any(row["expected"] == pytest.approx(mean, abs=0.01) for mean in of_three)
    # And that is what it reads.
    assert row["difference"] == pytest.approx(0.0, abs=1e-6)
    assert row["apart"] == pytest.approx(0.0, abs=1e-6)
    assert row["rank_correlation"] == pytest.approx(1.0)


def test_the_snr_estimate_is_set_against_the_ratio_white_noise_was_added_at(paths, monkeypatch):
    monkeypatch.setattr(checks, "SAMPLE", 5)
    monkeypatch.setattr(checks, "NOISE_SNRS_DB", (5.0, 15.0))
    store = save_clean_clips(paths, [spiky_clip(seed=number, seconds=4.0) for number in range(6)])
    row = checks.snr_against_noise(store)
    assert (row["name"], row["test_cases"]) == ("white noise by us", 5)  # a sample of the six
    # Each clip is given noise at a ratio of its own, from 5 dB to 15.
    assert 5.0 <= row["expected"] <= 15.0
    # The estimate reads these clips a little high, and a clip given less noise reads higher.
    assert 0.0 < row["difference"] < 4.0
    assert row["rank_correlation"] > 0.8


def test_an_experiment_says_what_to_run_when_the_audio_is_not_on_disk(paths):
    store = Store(paths)
    store.test_cases.write("urgent-clean", [TestCase("urgent2024_clean/a", audio_path="urgent2024_clean/a.wav")])
    with pytest.raises(RuntimeError, match="ba fetch --audio"):
        checks.level_against_gain(store)


def test_a_clip_passed_through_a_lower_rate_loses_what_is_above_half_of_it():
    clip = spiky_clip(seed=3)  # it holds as much at the top of its range as at the bottom
    cut = checks.resample_down_and_up(clip, RATE, 8000)
    assert len(cut) == len(clip)
    before = np.abs(np.fft.rfft(clip)) ** 2  # the energy at each frequency
    after = np.abs(np.fft.rfft(cut)) ** 2
    frequencies = np.fft.rfftfreq(len(clip), 1 / RATE)
    # Half of 8000 Hz is 4000. A little over it next to nothing is left, and a little under it nearly all is.
    assert after[frequencies > 4400].sum() < 0.001 * before[frequencies > 4400].sum()
    assert after[frequencies < 3600].sum() == pytest.approx(before[frequencies < 3600].sum(), rel=0.02)


def test_the_bandwidth_is_set_against_the_cut_of_each_lower_rate(paths, monkeypatch):
    monkeypatch.setattr(checks, "SAMPLE", 5)
    monkeypatch.setattr(checks, "LOWER_RATES_HZ", (4000, 8000))
    store = save_clean_clips(paths, [spiky_clip(seed=number) for number in range(6)])
    through_4000, through_8000 = checks.bandwidth_against_cut(store)
    # A row for each rate, in the order given, of a sample of the six. A pass through a rate cuts a clip off at half of it.
    assert (through_4000["name"], through_4000["test_cases"], through_4000["expected"]) == ("resampled to 4000 Hz", 5, 2000.0)
    assert (through_8000["name"], through_8000["test_cases"], through_8000["expected"]) == ("resampled to 8000 Hz", 5, 4000.0)
    # These clips hold as much at the top as anywhere, so each reads just over its cut, where the filter lets go.
    assert through_4000["measured"] == pytest.approx(1.05 * 2000.0, rel=0.01)
    assert through_8000["measured"] == pytest.approx(1.05 * 4000.0, rel=0.01)
    assert through_8000["difference"] == pytest.approx(through_8000["apart"])
    # Every clip of a row has the one cut, so the two readings cannot be ranked against each other.
    assert through_8000["rank_correlation"] is None


def test_a_clip_that_read_under_the_cut_before_is_left_out(paths, monkeypatch):
    monkeypatch.setattr(checks, "LOWER_RATES_HZ", (8000,))
    # Two clips that hold nothing above 1000 Hz, which a cut at 4000 Hz takes nothing from, and two that hold everything.
    narrow = [checks.resample_down_and_up(spiky_clip(seed=number), RATE, 2000) for number in range(2)]
    wide = [spiky_clip(seed=number) for number in range(2, 4)]
    store = save_clean_clips(paths, narrow + wide)
    (through_8000,) = checks.bandwidth_against_cut(store)
    assert through_8000["test_cases"] == 2


# ── the command ──────────────────────────────────────────────────────────────────────────


def test_the_command_runs_every_test(paths, monkeypatch):
    # URGENT's tags: the first three recordings have noise and nothing else, and the last three had their top cut off.
    tags_of = {}
    for number in range(6):
        tags_of[f"fileid_{number}"] = ["simulated;background_noise", "simulated;bandwidth_limitation;background_noise"][number // 3]
    paths = with_tags(paths, tags_of)
    monkeypatch.setattr(common, "get_paths", lambda: paths)
    store = Store(paths)

    # URGENT's two halves, six recordings each, with a second of audio on disk and what was measured on it.
    measured = {
        "urgent-clean": {
            "snr_est": [40.0, 41.0, 42.0, 43.0, 44.0, 45.0],
            "speech_fraction": [0.7, 0.8, 0.9, 0.4, 0.5, 0.6],
            "bandwidth_est": [3000.0, 5000.0, 7000.0, 6000.0, 7000.0, 8000.0],
        },
        "urgent-noisy": {
            "snr_given": [0.0, 5.0, 10.0, 3.0, 7.0, 14.0],
            "snr_est": [0.0, 4.0, 8.0, 1.0, 7.0, 10.0],
            "speech_fraction": [0.7, 0.8, 0.9, 0.2, 0.3, 0.4],
            "bandwidth_est": [3000.0, 5000.0, 7000.0, 3000.0, 3500.0, 4000.0],
        },
    }
    for benchmark_id, dataset in (("urgent-clean", "urgent2024_clean"), ("urgent-noisy", "urgent2024")):
        (paths.audio / dataset).mkdir(parents=True)
        test_cases = []
        for number in range(6):
            audio_path = f"{dataset}/fileid_{number}.flac"
            sf.write(str(paths.audio / audio_path), spiky_clip(seed=number), RATE)
            test_case = TestCase(f"{dataset}/fileid_{number}", audio_path=audio_path, duration=1.0, sampling_rate=RATE)
            for field, values in measured[benchmark_id].items():
                setattr(test_case, field, values[number])
            test_cases.append(test_case)
        store.test_cases.write(benchmark_id, test_cases)

    result = CliRunner().invoke(main.app, ["test_metadata"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    rows = [line.split() for line in result.output.splitlines()]

    def rows_of(*name: str) -> list[list[str]]:
        """The rows of the three comparisons that are of one thing done: the name, then a count of test cases."""
        return [row[len(name) :] for row in rows if tuple(row[: len(name)]) == name and row[len(name)].isdigit()]

    # A row is how many test cases, the mean of each reading, the difference, how far apart, and the rank
    # correlation. The estimate and the speech fraction are set against their other reading for each thing done.
    assert rows_of("noise", "only") == [
        ["3", "5.00", "4.00", "-1.00", "1.00", "1.00"],
        ["3", "0.800", "0.800", "0.000", "0.000", "1.00"],
    ]
    assert rows_of("cut", "off") == [
        ["3", "8.00", "6.00", "-2.00", "2.00", "1.00"],
        ["3", "0.500", "0.300", "-0.200", "0.200", "1.00"],
    ]
    assert rows_of("all") == [
        ["6", "6.50", "5.00", "-1.50", "1.50", "1.00"],
        ["6", "0.650", "0.550", "-0.100", "0.100", "1.00"],
    ]
    # The three experiments of our own are each on all six clean clips: the sample is no larger than the list.
    (louder_by_us,) = rows_of("6", "dB", "louder", "by", "us")
    (noise_by_us,) = rows_of("white", "noise", "by", "us")
    assert (louder_by_us[0], noise_by_us[0]) == ("6", "6")
    # A louder clip reads just that much more level.
    assert louder_by_us[3:] == ["0.00", "0.00", "1.00"]
    # The bandwidth has a row for each rate the clips are resampled to: how many clips, and the cut, which is half the rate.
    for lower_rate, cut_at in (("8000", "4000"), ("12000", "6000"), ("14000", "7000")):
        (resampled,) = rows_of("resampled", "to", lower_rate, "Hz")
        assert resampled[:2] == ["6", cut_at]
