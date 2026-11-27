"""Chunked sets: audio in chunks, one reference per session, scored per session as the clone does."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from ba.actions.scoring import score
from ba.config import load_config
from ba.fetching import benchmarks, manifests
from ba.normalize.registry import normalizers
from ba.store.store import Store
from ba.store.transcriptions import Run
from ba.utils.audio_utils import audio_file
from tests.conftest import leaderboards_score, tone_bytes

SESSIONS = {"call_a": "good morning everyone and welcome to the call", "call_b": "revenue rose twenty percent this quarter"}
CHUNKS = [  # id, parent, index, file
    ("uuid-a0", "call_a", 0, "call_a_chunk_0.wav"),
    ("uuid-a1", "call_a", 1, "call_a_chunk_1.wav"),
    ("uuid-b0", "call_b", 0, "call_b_chunk_0.wav"),
]


@pytest.fixture
def hub(tmp_path):
    """A local stand-in for the two Hub repos: an index of chunks, their audio, and the session texts."""
    root = tmp_path / "hub"
    (root / "chunked" / "audio").mkdir(parents=True)
    (root / "parent").mkdir(parents=True)
    with (root / "chunked" / "earnings22_cleaned_aa_chunked_v1.jsonl").open("w") as handle:
        for cid, parent, index, name in CHUNKS:
            handle.write(json.dumps({"id": cid, "parent_id": parent, "chunk_index": index, "duration": 0.4, "file_name": name}) + "\n")
            (root / "chunked" / "audio" / name).write_bytes(tone_bytes(seconds=0.4, fmt="WAV"))
    with (root / "parent" / "earnings22_cleaned_aa_v1.jsonl").open("w") as handle:
        for sid, text in SESSIONS.items():
            handle.write(json.dumps({"id": sid, "transcript": text}) + "\n")

    def fetch(repo, filename, revision=None):
        return root / ("chunked" if repo.endswith("chunked") else "parent") / filename

    def fetch_audio(repo, audio_dir, revision=None):
        return root / "chunked" / audio_dir

    def revision(repo):
        return "rev-chunked"

    return SimpleNamespace(fetch_file=fetch, fetch_audio=fetch_audio, revision=revision)


def pulled(paths, hub, with_audio=True):
    store, cfg = Store(paths), load_config(paths)
    test_cases = benchmarks.pull_benchmark(paths, store, cfg, "earnings22-cleaned", with_audio=with_audio, hub=hub)
    return store, cfg, test_cases


RUN = Run("fake/sys", "earnings22-cleaned")


def finished_run(paths, store, texts):
    """A model's transcript of every chunk, read from a transcript file as the leaderboard publishes one."""
    path = paths.data / "MODEL_fake-sys_DATASET_x__test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(reversed(public_rows(texts)))  # in no order: a chunk's place is what the row says it is
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifests.import_manifest(store, path, system_id=RUN.system_id, benchmark_id=RUN.benchmark_id)


def test_pull_stores_one_test_case_per_chunk_with_its_sessions_reference(paths, hub):
    store, _, pulled_test_cases = pulled(paths, hub)
    assert pulled_test_cases == 3
    test_cases = {c.test_case_id: c for c in store.test_cases.read("earnings22-cleaned")}
    first = test_cases["earnings22_cleaned_aa_chunked/uuid-a1"]
    assert first.parent_id == "call_a"
    assert first.chunk_index == 1 and first.duration == pytest.approx(0.4)
    assert (paths.audio / first.audio_path).is_file() and first.sampling_rate == 16000 and first.level is not None
    # A chunk carries its whole session's reference, so the fields counted in the reference are the session's.
    assert first.ref_words == len(SESSIONS["call_a"].split())
    # So is the speaking rate: the session's eight words over the length of its two chunks together.
    other = test_cases["earnings22_cleaned_aa_chunked/uuid-a0"]
    assert first.speaking_rate == other.speaking_rate == pytest.approx(8 / 0.8)
    assert test_cases["earnings22_cleaned_aa_chunked/uuid-b0"].speaking_rate == pytest.approx(6 / 0.4)
    # The list says its test cases are chunks of sessions, and a session has one reference.
    assert store.test_cases.scored_per_session("earnings22-cleaned")
    assert store.references.read("earnings22-cleaned") == {f"earnings22_cleaned_aa_chunked/{session}": text for session, text in SESSIONS.items()}


def test_scoring_joins_a_sessions_chunks_and_matches_the_clones_scorer(paths, hub):
    store, _, _ = pulled(paths, hub)
    texts = {"uuid-a0": "good morning everyone and", "uuid-a1": " welcome to the cool ", "uuid-b0": "revenue rose 20% this quarter"}
    finished_run(paths, store, texts)
    # A session's chunks are joined in order as they are read, as the leaderboard joins them: one transcript for each call.
    assert store.hypotheses.read(RUN) == {
        "earnings22_cleaned_aa_chunked/call_a": "good morning everyone and welcome to the cool",
        "earnings22_cleaned_aa_chunked/call_b": "revenue rose 20% this quarter",
    }
    normalizer = normalizers(paths).get("leaderboard")
    rows = score(store, RUN, normalizer)
    assert [r["test_case_id"] for r in rows] == ["earnings22_cleaned_aa_chunked/call_a", "earnings22_cleaned_aa_chunked/call_b"]
    assert (rows[0]["n_ref"], rows[0]["n_sub"]) == (8, 1)  # one reference per call, not per chunk
    assert rows[1]["n_sub"] + rows[1]["n_del"] + rows[1]["n_ins"] == 0
    # The leaderboard's own scorer gives the same totals for the same rows, in its file's form.
    theirs = leaderboards_score(paths, public_rows(texts))
    assert (theirs["sub"], theirs["del"], theirs["ins"]) == (1, 0, 0)


def test_a_chunk_with_no_text_leaves_its_sessions_other_chunks_standing(paths, hub):
    store, _, _ = pulled(paths, hub)
    finished_run(paths, store, {"uuid-a0": "good morning everyone and", "uuid-a1": "", "uuid-b0": ""})
    rows = score(store, RUN, normalizers(paths).get("leaderboard"))
    assert (rows[0]["empty"], rows[0]["n_del"]) == (False, 4)  # the second chunk's words are missing
    # "twenty percent" is one unit under the leaderboard's rules: 20%.
    assert (rows[1]["empty"], rows[1]["n_ref"], rows[1]["n_del"]) == (True, 5, 5)


def public_rows(texts: dict[str, str]) -> list[dict]:
    return [
        {"audio_filepath": cid, "duration": 0.4, "time": 0.01, "text": SESSIONS[parent], "pred_text": texts[cid], "parent_id": parent, "chunk_index": index}
        for cid, parent, index, _ in CHUNKS
    ]


def test_pull_without_audio_then_with_audio(paths, hub):
    store, _, _ = pulled(paths, hub, with_audio=False)
    assert all(audio_file(paths, c) is None and c.level is None for c in store.test_cases.read("earnings22-cleaned"))
    pulled(paths, hub)
    assert all(audio_file(paths, c) is not None and c.level is not None for c in store.test_cases.read("earnings22-cleaned"))
    shutil.rmtree(paths.audio)
