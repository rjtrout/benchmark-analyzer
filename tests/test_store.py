"""The store: its five tables, each with the same three verbs, and what removes them."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run
from tests.conftest import row

RUN = Run("oruk/orukqwen-asr-1.7b", "ami-cleaned")


def make_test_case(i: int, **more) -> TestCase:
    return TestCase(test_case_id=f"ami_cleaned/c{i}", duration=1.5 + i, **more)


# ── test cases ───────────────────────────────────────────────────────────────────────────


def test_a_benchmarks_test_cases_keep_their_order_and_their_metadata(paths):
    store = Store(paths)
    assert not store.test_cases.has("ami-cleaned")
    cases = [make_test_case(3, snr_given=5.0, ref_words=3, alphanumeric=True, speaking_rate=0.75), make_test_case(1), make_test_case(2, level=-20.5)]
    store.test_cases.write("ami-cleaned", cases)
    assert store.test_cases.has("ami-cleaned")
    assert store.test_cases.ids("ami-cleaned") == ["ami_cleaned/c3", "ami_cleaned/c1", "ami_cleaned/c2"]
    back = store.test_cases.read("ami-cleaned")
    assert back == cases  # every field is a column, and one that is not known yet comes back empty
    assert (back[0].duration, back[0].snr_given, back[0].alphanumeric, back[1].level, back[2].level) == (4.5, 5.0, True, None, -20.5)
    assert (back[0].speaking_rate, back[1].speaking_rate) == (0.75, None)


def test_saving_a_benchmarks_test_cases_again_replaces_them(paths):
    store = Store(paths)
    cases = [make_test_case(3), make_test_case(1), make_test_case(2)]
    store.test_cases.write("ami-cleaned", cases)
    # The same test cases again take what has been learned of them since.
    cases[0].level = -18.0
    store.test_cases.write("ami-cleaned", cases)
    assert store.test_cases.read("ami-cleaned")[0].level == -18.0
    # Other test cases replace them.
    store.test_cases.write("ami-cleaned", cases[:2])
    assert store.test_cases.ids("ami-cleaned") == ["ami_cleaned/c3", "ami_cleaned/c1"]


def test_a_list_says_whether_it_is_scored_session_by_session(paths):
    store = Store(paths)
    store.test_cases.write("cases", [make_test_case(1)])
    store.test_cases.write("calls", [make_test_case(1, parent_id="call-1", chunk_index=0), make_test_case(2, parent_id="call-1", chunk_index=1)])
    assert not store.test_cases.scored_per_session("cases")
    assert store.test_cases.scored_per_session("calls")  # its test cases are chunks of a session


def test_what_is_scored_is_a_test_case_or_the_session_it_is_a_chunk_of():
    assert make_test_case(1).unit_id == "ami_cleaned/c1"
    assert make_test_case(2, parent_id="call-1", chunk_index=0).unit_id == "ami_cleaned/call-1"
    assert make_test_case(3, parent_id="call-1", chunk_index=1).unit_id == "ami_cleaned/call-1"


def test_a_benchmark_with_no_list_says_how_to_get_one(paths):
    with pytest.raises(KeyError, match="ba fetch --benchmark ami-cleaned"):
        Store(paths).test_cases.ids("ami-cleaned")
    with pytest.raises(KeyError, match="ba fetch --benchmark ami-cleaned"):
        Store(paths).test_cases.read("ami-cleaned")
    with pytest.raises(KeyError, match="ba fetch --benchmark ami-cleaned"):
        Store(paths).test_cases.scored_per_session("ami-cleaned")


# ── references and transcripts ───────────────────────────────────────────────────────────


def test_a_run_is_a_model_on_a_benchmark_and_names_its_files():
    assert RUN.run_id == "oruk-orukqwen-asr-1.7b__ami-cleaned"
    assert Run("oruk/orukqwen-asr-1.7b", "ami-cleaned") == RUN


def test_a_models_transcripts_are_written_whole_and_read_back_by_test_case(paths):
    store = Store(paths)
    assert not store.hypotheses.has(RUN)
    # A test case the model has no transcript of has no row. One it wrote nothing for has an empty one.
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello there", "ami_cleaned/c2": ""})
    assert Store(paths).hypotheses.read(RUN) == {"ami_cleaned/c1": "hello there", "ami_cleaned/c2": ""}
    assert store.hypotheses.has(RUN)
    # A second fetch replaces the file: nothing is appended.
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello"})
    assert store.hypotheses.read(RUN) == {"ami_cleaned/c1": "hello"}


def test_the_runs_are_the_models_with_transcripts_benchmark_by_benchmark(paths):
    store = Store(paths)
    store.hypotheses.write(Run("b/model", "second"), {"x/1": "one"})
    store.hypotheses.write(Run("a/model", "second"), {"x/1": "one"})
    store.hypotheses.write(Run("a/model", "first"), {"y/1": "one"})
    runs = store.hypotheses.runs(["a/model", "b/model", "c/model"], ["first", "second", "third"])
    assert [(run.system_id, run.benchmark_id) for run in runs] == [("a/model", "first"), ("a/model", "second"), ("b/model", "second")]


def test_a_runs_transcripts_are_kept_as_written_and_as_each_normalizer_leaves_them(paths):
    store = Store(paths)
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "Hello, their!"})
    assert not store.hypotheses.has(RUN, "leaderboard")  # no normalizer has been applied yet
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello their"}, "leaderboard")
    assert store.hypotheses.has(RUN, "leaderboard") and not store.hypotheses.has(RUN, "verbatim")
    # Clean is the transcripts as written, and is what is read when no option is named.
    assert store.hypotheses.read(RUN) == store.hypotheses.read(RUN, "clean") == {"ami_cleaned/c1": "Hello, their!"}
    assert store.hypotheses.read(RUN, "leaderboard") == {"ami_cleaned/c1": "hello their"}
    # One folder for the run, with a file for each option, beside the references under transcriptions/.
    folder = paths.artifacts / "transcriptions" / "hypotheses" / RUN.run_id
    assert sorted(file.name for file in folder.iterdir()) == ["clean.parquet", "leaderboard.parquet"]


def test_a_benchmarks_references_are_kept_as_written_and_as_each_normalizer_leaves_them(paths):
    store = Store(paths)
    assert not store.references.has("ami-cleaned") and not store.references.has("ami-cleaned", "leaderboard")
    # Clean is the references as written, each under what is scored against it, and is what is read when no option is named.
    store.references.write("ami-cleaned", {"ami_cleaned/c1": "Reference number 1.", "ami_cleaned/c2": "Uh"})
    assert store.references.has("ami-cleaned")
    assert store.references.read("ami-cleaned") == store.references.read("ami-cleaned", "clean") == {
        "ami_cleaned/c1": "Reference number 1.",
        "ami_cleaned/c2": "Uh",
    }
    assert store.references.count("ami-cleaned") == 2
    # The references are a benchmark's: every model on it shares what a normalizer leaves of them.
    store.references.write("ami-cleaned", {"ami_cleaned/c1": "reference number one", "ami_cleaned/c2": ""}, "leaderboard")
    assert store.references.read("ami-cleaned", "leaderboard") == {"ami_cleaned/c1": "reference number one", "ami_cleaned/c2": ""}
    assert store.references.has("ami-cleaned", "leaderboard") and not store.references.has("ami-cleaned", "verbatim")
    # One folder for the benchmark, with a file for each option, as a run's transcripts have.
    folder = paths.artifacts / "transcriptions" / "references" / "ami-cleaned"
    assert sorted(file.name for file in folder.iterdir()) == ["clean.parquet", "leaderboard.parquet"]


def test_what_a_normalizer_left_can_be_forgotten_and_the_texts_as_written_stay(paths):
    store = Store(paths)
    store.references.write("ami-cleaned", {"ami_cleaned/c1": "Reference number 1."})
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "Hello, there!"})
    for normalizer_id in ("leaderboard", "verbatim"):
        store.references.write("ami-cleaned", {"ami_cleaned/c1": "reference number 1"}, normalizer_id)
        store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello there"}, normalizer_id)
    store.references.forget("ami-cleaned", "leaderboard")
    store.hypotheses.forget(RUN, "leaderboard")
    assert not store.references.has("ami-cleaned", "leaderboard") and not store.hypotheses.has(RUN, "leaderboard")
    # Another normalizer's texts, and the texts as written, are as they were.
    assert store.references.has("ami-cleaned", "verbatim") and store.hypotheses.has(RUN, "verbatim")
    assert store.references.read("ami-cleaned") == {"ami_cleaned/c1": "Reference number 1."}
    assert store.hypotheses.read(RUN) == {"ami_cleaned/c1": "Hello, there!"}
    # Forgetting what is not here is no error.
    store.references.forget("ami-cleaned", "leaderboard")
    store.hypotheses.forget(RUN, "leaderboard")


# ── scores ───────────────────────────────────────────────────────────────────────────────


def test_a_metric_is_kept_beside_the_others_and_one_not_computed_yet_is_blank(paths):
    store = Store(paths)
    assert not store.scores.has(RUN, "leaderboard", "wer")
    words = [row("ami_cleaned/c1", 4, n_sub=1), row("ami_cleaned/c2", 2, n_del=2, empty=True)]
    store.scores.write(RUN, "leaderboard", "wer", words)
    assert store.scores.has(RUN, "leaderboard", "wer")
    assert store.scores.read(RUN, "leaderboard", "wer") == words
    # Character errors and semantic error have their columns in the same file, blank until they are computed.
    assert not store.scores.has(RUN, "leaderboard", "cer") and not store.scores.has(RUN, "leaderboard", "semantic")
    # Another normalizer is another file.
    assert not store.scores.has(RUN, "verbatim", "wer")
    characters = [row("ami_cleaned/c1", 20, n_sub=2, n_ins=1), row("ami_cleaned/c2", 9, n_del=9, empty=True)]
    store.scores.write(RUN, "leaderboard", "cer", characters)
    assert store.scores.has(RUN, "leaderboard", "cer")
    assert store.scores.read(RUN, "leaderboard", "cer") == characters
    assert store.scores.read(RUN, "leaderboard", "wer") == words  # the word errors are as they were
    # One file for the run under the normalizer: a row per transcript, with the normalizer named in each.
    saved = pq.read_table(paths.artifacts / "scores" / "ours" / RUN.run_id / "leaderboard.parquet").to_pylist()
    assert [(r["test_case_id"], r["normalizer"], r["empty"], r["wer_ref"], r["wer_sub"], r["cer_ref"], r["cer_ins"], r["semantic"]) for r in saved] == [
        ("ami_cleaned/c1", "leaderboard", False, 4, 1, 20, 1, None),
        ("ami_cleaned/c2", "leaderboard", True, 2, 0, 9, 0, None),
    ]


def test_semantic_error_is_kept_as_a_rate_and_read_with_the_reference_words_as_its_weight(paths):
    store = Store(paths)
    store.scores.write(RUN, "leaderboard", "wer", [row("ami_cleaned/c1", 4, n_sub=1), row("ami_cleaned/c2", 2, n_del=2, empty=True)])
    rates = [row("ami_cleaned/c1", 4, rate=0.25), row("ami_cleaned/c2", 2, empty=True, rate=1.0)]
    store.scores.write(RUN, "leaderboard", "semantic", rates)
    assert store.scores.has(RUN, "leaderboard", "semantic")
    # A row of semantic error holds the rate, and no counts. Its weight is the reference's words, which the word errors hold.
    assert store.scores.read(RUN, "leaderboard", "semantic") == rates
    assert store.scores.read(RUN, "leaderboard", "wer")[0] == row("ami_cleaned/c1", 4, n_sub=1)


def test_a_metric_written_again_replaces_its_figures_test_case_by_test_case(paths):
    store = Store(paths)
    store.scores.write(RUN, "leaderboard", "wer", [row("ami_cleaned/c1", 4, n_sub=1), row("ami_cleaned/c2", 2)])
    # The rows are matched by what they are of, whatever order they come in, and the file keeps its order.
    store.scores.write(RUN, "leaderboard", "wer", [row("ami_cleaned/c2", 2, n_ins=3), row("ami_cleaned/c1", 4)])
    assert store.scores.read(RUN, "leaderboard", "wer") == [row("ami_cleaned/c1", 4), row("ami_cleaned/c2", 2, n_ins=3)]


def test_the_leaderboards_own_score_is_kept_beside_ours(paths):
    store = Store(paths)
    assert not store.published_scores.has(RUN)
    store.published_scores.write(RUN, {"wer": 7.55, "wer_sub": 2938, "wer_del": 1853, "wer_ins": 1423})
    assert store.published_scores.has(RUN)
    assert store.published_scores.read(RUN) == {"wer": 7.55, "wer_sub": 2938, "wer_del": 1853, "wer_ins": 1423}
    assert (paths.artifacts / "scores" / "huggingface" / RUN.run_id / "leaderboard.parquet").exists()


# ── writing ──────────────────────────────────────────────────────────────────────────────


def test_a_file_being_written_is_read_as_it_was_until_it_is_whole(paths, monkeypatch):
    store = Store(paths)
    kept_meanwhile = []  # what a reader finds kept, at the moment the last byte of a file is written
    write_table = pq.write_table

    def write_and_look(table, where, **options):
        write_table(table, where, **options)
        kept_meanwhile.append((store.scores.has(RUN, "leaderboard", "wer"), store.scores.has(RUN, "leaderboard", "cer")))

    monkeypatch.setattr(pq, "write_table", write_and_look)
    store.scores.write(RUN, "leaderboard", "wer", [row("ami_cleaned/c1", 4, n_sub=1)])
    store.scores.write(RUN, "leaderboard", "cer", [row("ami_cleaned/c1", 20, n_sub=2)])
    # A page of the web UI, or another command, may read the store while a file is written. It finds no file
    # until the first is whole, and then the first, with no character errors, until the second is whole.
    assert kept_meanwhile == [(False, False), (True, False)]
    assert store.scores.has(RUN, "leaderboard", "wer") and store.scores.has(RUN, "leaderboard", "cer")


def test_a_write_that_is_stopped_leaves_the_file_as_it_was_and_nothing_beside_it(paths, monkeypatch):
    store = Store(paths)
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello"})
    write_table = pq.write_table

    def write_and_stop(table, where, **options):
        write_table(table, where, **options)
        raise KeyboardInterrupt  # Ctrl-C, as the last byte is written

    monkeypatch.setattr(pq, "write_table", write_and_stop)
    with pytest.raises(KeyboardInterrupt):
        store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello there"})
    assert store.hypotheses.read(RUN) == {"ami_cleaned/c1": "hello"}
    folder = paths.artifacts / "transcriptions" / "hypotheses" / RUN.run_id
    assert [file.name for file in folder.iterdir()] == ["clean.parquet"]


# ── removing ─────────────────────────────────────────────────────────────────────────────


def test_everything_stored_can_be_removed_at_once_as_a_fetch_does(paths):
    store = Store(paths)
    store.test_cases.write("ami-cleaned", [make_test_case(1)])
    store.references.write("ami-cleaned", {"ami_cleaned/c1": "reference number 1"})
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello"})
    store.published_scores.write(RUN, {"wer": 1.0, "wer_sub": 0, "wer_del": 0, "wer_ins": 0})
    store.scores.write(RUN, "leaderboard", "wer", [row("ami_cleaned/c1", 1)])
    store.remove_everything()
    assert not paths.artifacts.exists()
    assert not store.test_cases.has("ami-cleaned") and not store.references.has("ami-cleaned")
    assert store.hypotheses.runs([RUN.system_id], ["ami-cleaned"]) == []
    store.remove_everything()  # nothing is stored: there is nothing to remove, and no error


def test_what_was_worked_out_here_can_be_cleared_and_what_was_fetched_stays(paths):
    store = Store(paths)
    store.test_cases.write("ami-cleaned", [make_test_case(1)])
    store.references.write("ami-cleaned", {"ami_cleaned/c1": "reference number 1"})
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello"})
    store.published_scores.write(RUN, {"wer": 1.0, "wer_sub": 0, "wer_del": 0, "wer_ins": 0})
    store.references.write("ami-cleaned", {"ami_cleaned/c1": "hello"}, "leaderboard")
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello"}, "leaderboard")
    store.scores.write(RUN, "leaderboard", "wer", [row("ami_cleaned/c1", 1)])
    store.clear("scores")
    assert not store.scores.has(RUN, "leaderboard", "wer") and not store.references.has("ami-cleaned", "leaderboard")
    assert not store.hypotheses.has(RUN, "leaderboard")
    # Of the texts, only those as written are left: the references and the transcripts.
    assert sorted(str(file.relative_to(paths.artifacts)) for file in (paths.artifacts / "transcriptions").rglob("*.parquet")) == [
        f"transcriptions/hypotheses/{RUN.run_id}/clean.parquet",
        "transcriptions/references/ami-cleaned/clean.parquet",
    ]
    assert store.references.read("ami-cleaned") == {"ami_cleaned/c1": "reference number 1"}
    # The list, the transcripts as written and the leaderboard's own score came from outside, and stay.
    assert store.test_cases.has("ami-cleaned") and store.hypotheses.read(RUN) == {"ami_cleaned/c1": "hello"} and store.published_scores.has(RUN)
    # A download is a folder under data/.
    (paths.data / "audio" / "ami_cleaned").mkdir(parents=True)
    (paths.data / "audio" / "ami_cleaned" / "c1.flac").write_bytes(b"sound")
    store.clear("audio")
    assert not (paths.data / "audio").exists()
    store.clear("transcript-files")  # nothing of that kind is here: there is nothing to remove, and no error
    with pytest.raises(ValueError, match="scores, leaderboard-scores, test-cases, audio, transcript-files"):
        store.clear("lists")
    assert store.test_cases.has("ami-cleaned")


def test_clearing_all_removes_every_category(paths):
    store = Store(paths)
    store.test_cases.write("ami-cleaned", [make_test_case(1)])
    store.references.write("ami-cleaned", {"ami_cleaned/c1": "reference number 1"})
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello"})
    store.hypotheses.write(RUN, {"ami_cleaned/c1": "hello"}, "leaderboard")
    store.published_scores.write(RUN, {"wer": 1.0, "wer_sub": 0, "wer_del": 0, "wer_ins": 0})
    store.scores.write(RUN, "leaderboard", "wer", [row("ami_cleaned/c1", 1)])
    paths.audio.mkdir(parents=True)
    paths.leaderboard.mkdir(parents=True)
    store.clear("all")
    assert not store.scores.has(RUN, "leaderboard", "wer") and not store.hypotheses.has(RUN, "leaderboard")
    assert not store.published_scores.has(RUN) and not store.test_cases.has("ami-cleaned")
    assert not paths.audio.exists() and not paths.leaderboard.exists()
    # The texts as written are no category of their own: only a fetch replaces them.
    assert store.hypotheses.read(RUN) == {"ami_cleaned/c1": "hello"}
