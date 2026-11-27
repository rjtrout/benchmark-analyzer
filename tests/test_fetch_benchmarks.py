from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from ba.config import load_config
from ba.fetching import benchmarks
from ba.fetching.loaders import ParquetSource
from ba.store.store import Store
from ba.utils.audio_utils import audio_file
from tests.conftest import tone_bytes

ROWS = [
    ("fileid_3", "The third test case.", 0.5, 12.5),
    ("fileid_1", "Uh", 0.25, -3.0),
    ("fileid_2", "Twenty two apples", 0.75, 5.0),
]


def hub_like_parquet(tmp_path, rows=ROWS, name="test-00000-of-00001.parquet"):
    """A Parquet file laid out as the Hub stores an audio dataset."""
    path = tmp_path / "hub" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "dataset": ["urgent2024_clean"] * len(rows),
            "audio": [{"bytes": tone_bytes(seconds=r[2], freq=300 + 100 * i), "path": f"{r[0]}.flac"} for i, r in enumerate(rows)],
            "text": [r[1] for r in rows],
            "id": [r[0] for r in rows],
            "audio_length_s": pa.array([r[2] for r in rows], type=pa.float32()),
            "snr_db": [r[3] for r in rows],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def source_for(tmp_path, **kwargs):
    return ParquetSource(repo="hf-audio/open-asr-leaderboard", revision="rev123", files=[str(hub_like_parquet(tmp_path, **kwargs))])


# ── pulling test cases and saving the list ───────────────────────────────────────────────


def test_pull_stores_test_cases_and_audio_and_saves_the_list_in_file_order(paths, tmp_path):
    store, cfg = Store(paths), load_config(paths)
    assert benchmarks.pull_benchmark(paths, store, cfg, "urgent-clean", source=source_for(tmp_path), with_audio=True) == 3
    assert store.test_cases.ids("urgent-clean") == [
        "urgent2024_clean/fileid_3", "urgent2024_clean/fileid_1", "urgent2024_clean/fileid_2",
    ]  # fmt: skip
    third, first, second = store.test_cases.read("urgent-clean")
    assert (first.test_case_id, first.parent_id) == ("urgent2024_clean/fileid_1", None)
    # The references as written are saved beside the list, in its order, and nowhere else.
    assert store.references.read("urgent-clean") == {
        "urgent2024_clean/fileid_3": "The third test case.",
        "urgent2024_clean/fileid_1": "Uh",
        "urgent2024_clean/fileid_2": "Twenty two apples",
    }
    # What the source gives is read into the test case, and the audio's own rate once the audio is here.
    assert first.duration == pytest.approx(0.25) and first.snr_given == -3.0 and first.sampling_rate == 16000
    # The fields counted in the reference are counted as it is read.
    assert [(c.ref_words, c.alphanumeric) for c in (third, first, second)] == [(4, False), (1, False), (3, True)]
    # And the speaking rate from the two: the reference's words over the stated length.
    assert [c.speaking_rate for c in (third, first, second)] == pytest.approx([4 / 0.5, 1 / 0.25, 3 / 0.75])
    # The fields measured on the audio are measured as it is fetched.
    assert first.level < 0 and first.snr_est is not None and first.bandwidth_est > 0 and 0 <= first.speech_fraction <= 1
    stored = paths.audio / first.audio_path
    assert sf.info(stored).duration == pytest.approx(0.25)


def test_a_pull_fetches_the_list_alone_and_a_later_pull_can_add_the_audio(paths, tmp_path):
    store, cfg = Store(paths), load_config(paths)
    assert benchmarks.pull_benchmark(paths, store, cfg, "urgent-clean", source=source_for(tmp_path)) == 3
    # Without the audio, what the audio would say is empty. What the source states and the reference holds is there.
    listed = store.test_cases.read("urgent-clean")
    assert all(c.audio_path is None and c.sampling_rate is None and c.level is None for c in listed)
    assert [c.duration for c in listed] == [0.5, 0.25, 0.75] and [c.ref_words for c in listed] == [4, 1, 3]
    benchmarks.pull_benchmark(paths, store, cfg, "urgent-clean", source=source_for(tmp_path), with_audio=True)
    with_audio = store.test_cases.read("urgent-clean")
    assert [c.test_case_id for c in with_audio] == [c.test_case_id for c in listed]
    assert all(audio_file(paths, c) is not None and c.sampling_rate == 16000 and c.level is not None for c in with_audio)


def test_ids_are_kept_out_of_the_way_of_the_file_system(paths, tmp_path):
    rows = [("a/b c?.wav", "one", 0.2, 1.0), ("a_b c_.wav", "two", 0.2, 1.0)]
    store, cfg = Store(paths), load_config(paths)
    benchmarks.pull_benchmark(paths, store, cfg, "urgent-clean", source=source_for(tmp_path, rows=rows), with_audio=True)
    files = [c.audio_path for c in store.test_cases.read("urgent-clean")]
    assert len(set(files)) == 2
    assert all((paths.audio / f).is_file() for f in files)
