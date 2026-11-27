"""Metadata: what the source states of a test case, what is derived here, and filling both in."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from ba.metadata.derived import DERIVED, FROM_AUDIO, get_derived
from ba.metadata.given import GIVEN, get_given
from ba.metadata.populate import populate_metadata, populate_speaking_rate
from ba.store.test_cases import TestCase
from tests.conftest import tone_bytes


def test_every_field_is_a_column_of_the_test_cases():
    assert set(GIVEN) | set(DERIVED) <= set(TestCase.__dataclass_fields__)


# ── given ────────────────────────────────────────────────────────────────────────────────


def test_what_the_source_states_is_read_from_the_test_cases_row():
    # The leaderboard's sets state a length as audio_length_s, and URGENT's noisy half the level its noise was mixed at.
    assert get_given({"audio_length_s": 3.5, "snr_db": 4.5}, None) == {"duration": 3.5, "snr_given": 4.5}
    # Earnings22's chunks state their length as duration. No other set states an SNR, and what is not stated is left out.
    assert get_given({"duration": 26.9, "snr_db": None}, None) == {"duration": 26.9}


def test_the_sampling_rate_is_the_audio_files_own_and_the_stated_length_is_taken_as_true(tmp_path):
    audio = tmp_path / "a.flac"
    audio.write_bytes(tone_bytes(seconds=4.0, sr=8000))
    # The file is four seconds long. What the source states is not checked against it.
    assert get_given({"audio_length_s": 5.0}, audio) == {"duration": 5.0, "sampling_rate": 8000}


# ── derived ──────────────────────────────────────────────────────────────────────────────


def test_two_fields_are_counted_in_the_reference(paths):
    counted = {}
    for reference in ("Yeah, I can't see 20 of them.", "Is that the right one?", "Twenty five percent, no more"):
        derived = get_derived(reference, None, paths)
        counted[reference] = (derived["ref_words"], derived["alphanumeric"])
        # The four measured on the audio are left out until the audio is fetched: nothing stored of them is touched.
        assert set(derived) == {"ref_words", "alphanumeric"}
    assert counted == {
        "Yeah, I can't see 20 of them.": (7, True),
        "Is that the right one?": (5, False),  # "the right one" is not a number
        "Twenty five percent, no more": (5, True),
    }


def tone_file(folder, name: str, amplitude: float):
    """A half-second tone on disk, at some share of full amplitude."""
    samples, rate = sf.read(io.BytesIO(tone_bytes(seconds=0.5)), dtype="float64")
    path = folder / name
    sf.write(str(path), samples * amplitude, rate)
    return path


def test_four_fields_are_measured_on_the_audio(tmp_path, paths):
    quiet = get_derived("alpha beta", tone_file(tmp_path, "quiet.flac", 0.5), paths)
    loud = get_derived("alpha beta", tone_file(tmp_path, "loud.flac", 1.0), paths)
    assert loud["level"] - quiet["level"] == pytest.approx(6.02, abs=0.01)  # twice the amplitude is 6 dB more
    assert all(isinstance(loud[field], float) for field in FROM_AUDIO)
    assert 0.0 <= loud["speech_fraction"] <= 1.0 and 0.0 < loud["bandwidth_est"] <= 8000.0  # half the file's 16 kHz
    assert (loud["ref_words"], loud["alphanumeric"]) == (2, False)  # the two counted in the reference, as before


def test_a_file_of_two_channels_is_measured_as_their_mean(tmp_path, paths):
    samples, rate = sf.read(io.BytesIO(tone_bytes(seconds=0.5)), dtype="float64")
    one_channel = tmp_path / "one.flac"
    two_channels = tmp_path / "two.flac"
    sf.write(str(one_channel), samples, rate)
    sf.write(str(two_channels), np.column_stack([samples, samples]), rate)
    # The same tone in both channels: their mean is the tone itself. Their sum would read 6 dB more.
    level = get_derived("alpha beta", one_channel, paths)["level"]
    assert get_derived("alpha beta", two_channels, paths)["level"] == pytest.approx(level, abs=0.01)


# ── both, filled into a test case ────────────────────────────────────────────────────────


def test_populating_fills_a_test_case_with_what_is_given_and_then_what_is_derived(paths):
    test_case = TestCase("urgent2024/f1")
    row = {"audio_length_s": 5.0, "snr_db": 4.5}
    populate_metadata(test_case, row, "Twenty two apples", paths)
    assert (test_case.duration, test_case.snr_given, test_case.ref_words, test_case.alphanumeric) == (5.0, 4.5, 3, True)
    # Its audio is not fetched: what the audio would say is still empty.
    assert (test_case.sampling_rate, test_case.level, test_case.snr_est) == (None, None, None)

    # With the audio on disk, the rest is filled too.
    (paths.audio / "urgent2024").mkdir(parents=True)
    tone_file(paths.audio / "urgent2024", "f1.flac", 0.5)
    test_case.audio_path = "urgent2024/f1.flac"
    populate_metadata(test_case, row, "Twenty two apples", paths)
    assert test_case.sampling_rate == 16000 and test_case.level < 0 and test_case.speech_fraction is not None
    assert (test_case.duration, test_case.ref_words) == (5.0, 3)


# ── the speaking rate, filled over a benchmark's whole list ──────────────────────────────


def test_the_speaking_rate_is_the_references_words_over_the_test_cases_length(paths):
    quick, slow, unstated = TestCase("urgent2024/f1"), TestCase("urgent2024/f2"), TestCase("urgent2024/f3")
    populate_metadata(quick, {"audio_length_s": 2.0}, "one two three four five", paths)
    populate_metadata(slow, {"audio_length_s": 8.0}, "Yeah, okay.", paths)
    populate_metadata(unstated, {}, "one two three", paths)
    # Filling one test case leaves it empty: a session's chunks share theirs, so it is worked out over the list.
    assert quick.speaking_rate is None
    populate_speaking_rate([quick, slow, unstated])
    # The words as written, over the whole length: a pause counts as length.
    assert (quick.speaking_rate, slow.speaking_rate) == (2.5, 0.25)
    assert unstated.speaking_rate is None  # no length is stated, so there is no rate to give


def test_a_sessions_chunks_share_the_sessions_speaking_rate(paths):
    # A chunk carries its whole session's reference, so its words are the session's. So is the length: the session's chunks together.
    call = "good morning everyone and welcome to the call"
    chunks = [TestCase(f"calls/a{index}", parent_id="call_a", chunk_index=index) for index in range(2)]
    other = TestCase("calls/b0", parent_id="call_b", chunk_index=0)
    populate_metadata(chunks[0], {"duration": 1.0}, call, paths)
    populate_metadata(chunks[1], {"duration": 3.0}, call, paths)
    populate_metadata(other, {"duration": 2.0}, "revenue rose", paths)
    populate_speaking_rate([*chunks, other])
    assert [chunk.speaking_rate for chunk in chunks] == [2.0, 2.0]  # eight words in four seconds, for both
    assert other.speaking_rate == 1.0  # another session has its own
