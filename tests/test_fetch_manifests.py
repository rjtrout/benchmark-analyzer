"""The leaderboard's transcript files: reading one, and joining it to a benchmark's test cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ba.actions.scoring import score
from ba.fetching import leaderboard, manifests
from ba.normalize.registry import normalizers
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run

REFS = {"c1": "Hello there, world.", "c2": "Uh", "c3": "Twenty two apples", "c4": "see you soon"}
RUN = Run("s/m", "ami-cleaned")


def listed_store(paths, refs: dict[str, str] = REFS, benchmark: str = "ami-cleaned", key: str = "ami_cleaned") -> Store:
    store = Store(paths)
    store.test_cases.write(benchmark, [TestCase(f"{key}/{sid}", duration=2.0) for sid in refs])
    store.references.write(benchmark, {f"{key}/{sid}": ref for sid, ref in refs.items()})
    return store


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def row(name: str, text: str, pred: str, **more) -> dict:
    return {"audio_filepath": name, "duration": 2.0, "time": 0.05, "text": text, "pred_text": pred, **more}


def manifest_rows(preds: dict[str, str]) -> list[dict]:
    return [row(sid, REFS[sid], pred) for sid, pred in preds.items()]


def imported(paths, store, rows, benchmark="ami-cleaned", system="s/m", name="m.jsonl") -> int:
    path = write_rows(paths.data / name, rows)
    return manifests.import_manifest(store, path, system_id=system, benchmark_id=benchmark)


def transcripts(store, run=RUN) -> dict[str, str]:
    """A run's stored transcripts, by each test case's id in its dataset."""
    return {test_case_id.split("/", 1)[1]: text for test_case_id, text in store.hypotheses.read(run).items()}


def test_a_manifests_file_name_gives_its_model_and_its_set():
    name = "MODEL_oruk-orukqwen-asr-1.7b_DATASET_hf-audio-open-asr-leaderboard_ami_cleaned_test.jsonl"
    assert manifests.parse_manifest_filename(name) == ("oruk-orukqwen-asr-1.7b", "hf-audio-open-asr-leaderboard_ami_cleaned_test")
    assert manifests.model_folder("oruk/orukqwen-asr-1.7b") == "oruk-orukqwen-asr-1.7b"


def test_import_joins_each_row_to_its_test_case_and_a_test_case_with_no_row_has_no_transcript(paths):
    store = listed_store(paths)
    # The leaderboard's runner drops c2 before the model runs: "Uh" normalizes to nothing. The file has no row for it.
    assert imported(paths, store, manifest_rows({"c1": "hello there world", "c3": "22 apples", "c4": ""}), system="oruk/orukqwen-asr-1.7b") == 3
    run = Run("oruk/orukqwen-asr-1.7b", "ami-cleaned")
    assert store.hypotheses.runs(["oruk/orukqwen-asr-1.7b"], ["ami-cleaned"]) == [run]
    # A model that was given a test case and wrote nothing has an empty transcript of it.
    assert transcripts(store, run) == {"c1": "hello there world", "c3": "22 apples", "c4": ""}
    assert len(score(store, run, normalizers(paths).get("leaderboard"))) == 3


def test_a_row_for_a_test_case_that_is_not_on_the_list_is_left_out(paths):
    store = listed_store(paths)
    # One row for four test cases, and one row for a test case that is not on the list.
    assert imported(paths, store, manifest_rows({"c1": "hello there world"}) + [row("c9", "not on the list", "x")]) == 1
    assert transcripts(store) == {"c1": "hello there world"}
    assert len(score(store, RUN, normalizers(paths).get("leaderboard"))) == 1


# ── rows that carry less than a full test case ID ────────────────────────────────────────

SPGI = {"call1/1.wav": "good morning everyone", "call2/1.wav": "thank you", "call3/1.wav": "thank you", "call3/2.wav": "next question please"}
SPGI_RUN = Run("s/m", "spgispeech")


def test_rows_named_by_file_name_only_are_told_apart_by_their_reference(paths):
    # SPGISpeech's test case IDs are <call>/<number>.wav, and most runners write the file name alone.
    store = listed_store(paths, SPGI, "spgispeech", "spgispeech")
    rows = [row("2.wav", "next question please", "next question"), row("1.wav", "good morning everyone", "morning all"), row("1.wav", "thank you", "thanks one"), row("1.wav", "thank you", "thanks two")]
    assert imported(paths, store, rows, benchmark="spgispeech") == 4
    hyps = transcripts(store, SPGI_RUN)
    assert hyps["call1/1.wav"] == "morning all" and hyps["call3/2.wav"] == "next question"
    # Two test cases with one file name and one reference: taken in the order of the list.
    assert (hyps["call2/1.wav"], hyps["call3/1.wav"]) == ("thanks one", "thanks two")
    # The transcripts are kept in the list's order, whatever the file's order was.
    assert list(hyps) == list(SPGI)


def test_a_row_with_the_full_id_is_matched_by_it_alone(paths):
    store = listed_store(paths, SPGI, "spgispeech", "spgispeech")
    rows = [row(sid, ref, f"pred for {sid}") for sid, ref in SPGI.items()]
    rows[2]["text"] = "Thank you."  # the full ID still finds its test case, whatever the row's reference says
    assert imported(paths, store, rows, benchmark="spgispeech") == 4
    assert transcripts(store, SPGI_RUN)["call3/1.wav"] == "pred for call3/1.wav"


def test_a_file_name_shared_by_test_cases_with_other_references_is_left_unmatched(paths):
    store = listed_store(paths, SPGI, "spgispeech", "spgispeech")
    assert imported(paths, store, [row("1.wav", "a reference no test case has", "x")], benchmark="spgispeech") == 0


def test_chunks_are_joined_by_session_and_position_whatever_the_file_is_called(paths):
    store = Store(paths)
    # The first call's chunks are listed out of order: a chunk's place in its call is what the list says it is.
    chunks = [("uuid-a1", "call_a", 1), ("uuid-a0", "call_a", 0), ("uuid-b0", "call_b", 0)]
    test_cases = [TestCase(f"e22/{cid}", parent_id=parent, chunk_index=index, duration=2.0) for cid, parent, index in chunks]
    store.test_cases.write("earnings22-cleaned", test_cases)
    store.references.write("earnings22-cleaned", {"e22/call_a": "reference of call_a", "e22/call_b": "reference of call_b"})
    # One runner names a chunk by its ID, another by its file. Both say which session and which position.
    rows = [
        row("call_b_chunk_0.wav", "reference of call_b", "b zero", parent_id="call_b", chunk_index=0),
        row("call_a_chunk_1.wav", "reference of call_a", "a one", parent_id="call_a", chunk_index=1),
        row("uuid-a0", "reference of call_a", "a zero", parent_id="call_a", chunk_index=0),
    ]
    assert imported(paths, store, rows, benchmark="earnings22-cleaned") == 3
    # Each row found its chunk, and a session's chunks are joined in order into the session's one transcript.
    assert transcripts(store, Run("s/m", "earnings22-cleaned")) == {"call_a": "a zero a one", "call_b": "b zero"}


# ── the score file beside a manifest ─────────────────────────────────────────────────────


def test_import_keeps_the_leaderboards_own_score_that_was_written_for_this_file(paths):
    store = listed_store(paths)
    path = write_rows(paths.data / "m.jsonl", manifest_rows({"c1": "a", "c3": "b", "c4": "c"}))
    score = {"wer": 12.5, "ins": 1, "del": 2, "sub": 3, "rtfx": 100.0, "manifest_sha256": manifests.file_sha256(path)}
    manifests.score_file_for(path).write_text(json.dumps(score))
    manifests.import_manifest(store, path, system_id="s/m", benchmark_id="ami-cleaned")
    # It is kept as a score, beside ours: its WER and the three error counts behind it.
    assert store.published_scores.read(RUN) == {"wer": 12.5, "wer_sub": 3, "wer_del": 2, "wer_ins": 1}


def test_a_score_file_written_for_another_file_is_not_kept(paths):
    store = listed_store(paths)
    path = write_rows(paths.data / "m.jsonl", manifest_rows({"c1": "a", "c3": "b", "c4": "c"}))
    manifests.score_file_for(path).write_text(json.dumps({"wer": 12.5, "ins": 1, "del": 2, "sub": 3, "manifest_sha256": "0" * 64}))
    manifests.import_manifest(store, path, system_id="s/m", benchmark_id="ami-cleaned")
    assert not store.published_scores.has(RUN)
    # And a file with no score beside it has none.
    imported(paths, store, manifest_rows({"c1": "x", "c3": "y", "c4": "z"}), system="another/model", name="other.jsonl")
    assert not store.published_scores.has(Run("another/model", "ami-cleaned"))


# ── the public bucket ────────────────────────────────────────────────────────────────────


@dataclass
class FakeFile:
    path: str
    size: int = 10
    uploaded_at: datetime | None = None


class FakeBucketApi:
    """Stands in for huggingface_hub.HfApi. Serves files from a dict."""

    def __init__(self, files: dict[str, str], uploaded_at: datetime | None = None):
        self.files = files
        self.uploaded_at = uploaded_at or datetime(2026, 11, 2, 14, 14, tzinfo=UTC)
        self.downloaded: list[str] = []

    def list_bucket_tree(self, bucket_id, prefix=None, recursive=None, token=None):
        assert bucket_id == leaderboard.PUBLIC_BUCKET and token is False  # public: no account is used
        return [FakeFile(p, len(c.encode()), self.uploaded_at) for p, c in self.files.items() if p.startswith(prefix or "")]

    def download_bucket_files(self, bucket_id, files, raise_on_missing_files=False, token=None):
        assert token is False
        for source, target in files:
            key = source if isinstance(source, str) else source.path
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_text(self.files[key], encoding="utf-8")
            self.downloaded.append(key)


def bucket_fixture() -> dict[str, str]:
    folder = "oruk-orukqwen-asr-1.7b"
    prefix = f"{folder}/MODEL_{folder}_DATASET_hf-audio-open-asr-leaderboard_"
    return {
        f"{prefix}ami_cleaned_test.jsonl": '{"audio_filepath": "c1", "duration": 1, "time": 1, "text": "hello", "pred_text": "hello"}\n',
        f"{prefix}ami_cleaned_test.score.json": "{}",
        f"{prefix}spgispeech_test.jsonl": "{}\n",
        f"{prefix}spgispeech_test.score.json": "{}",
        f"{prefix}librispeech_test.clean.jsonl": "{}\n",
        f"{folder}/longform/MODEL_{folder}_DATASET_hf-audio-open-asr-leaderboard_ami_cleaned_test.jsonl": "{}\n",
        f"{folder}/notes.txt": "x",
    }


AMI = "hf-audio-open-asr-leaderboard_ami_cleaned_test"


def test_pull_fetches_only_the_manifests_asked_for(paths):
    api = FakeBucketApi(bucket_fixture())
    got = leaderboard.pull_manifests(paths, "oruk/orukqwen-asr-1.7b", [AMI], api=api)
    assert [(f.path.name, f.dataset_id) for f in got] == [(f"MODEL_oruk-orukqwen-asr-1.7b_DATASET_{AMI}.jsonl", AMI)]
    assert got[0].path.is_file() and got[0].path.parent == paths.leaderboard / "hf-audio-asr_leaderboard_h200" / "oruk-orukqwen-asr-1.7b"
    assert got[0].uploaded_at == "2026-11-02T14:14:00+00:00"
    # The score file comes along, and nothing else does: not another set, and not a file in a subfolder.
    assert sorted(p.split("_DATASET_")[1] for p in api.downloaded) == [f"{AMI}.jsonl", f"{AMI}.score.json"]


def test_a_file_that_is_here_already_is_downloaded_again_all_the_same(paths):
    api = FakeBucketApi(bucket_fixture())
    leaderboard.pull_manifests(paths, "oruk/orukqwen-asr-1.7b", [AMI], api=api)
    again = FakeBucketApi(bucket_fixture())
    got = leaderboard.pull_manifests(paths, "oruk/orukqwen-asr-1.7b", [AMI], api=again)
    # The folder is there from the first time. The transcript file and its score file are fetched anew.
    assert len(again.downloaded) == 2 and got[0].path.is_file()


def test_a_model_with_no_transcripts_for_a_set_gives_nothing(paths):
    api = FakeBucketApi(bucket_fixture())
    assert leaderboard.pull_manifests(paths, "oruk/orukqwen-asr-1.7b", ["VoiceArena-Monsoon_en_IN_test__test"], api=api) == []
    assert leaderboard.pull_manifests(paths, "nobody/no-model", [AMI], api=api) == []
