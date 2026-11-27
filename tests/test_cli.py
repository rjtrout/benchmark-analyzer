"""The command line, end to end on a scratch project. No network and no account."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ba.cli import common, main
from ba.fetching import benchmarks, leaderboard, manifests
from ba.metadata.populate import populate_metadata
from ba.paths import Paths
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run
from tests.conftest import REPO_ROOT

runner = CliRunner()

SYSTEMS = """
systems:
  - id: lab/alpha
    published: { toy-cases: 20.00, toy-calls: 12.50 }
  - id: lab/beta
    published: { toy-cases: 99.99 }
"""
BENCHMARKS = """
benchmarks:
  - id: toy-cases
    loader: hf
    dataset_key: toy_cases
    source: { repo: org/toy, config: cases, split: test }
    manifest_aliases: [org-toy_old-name_test]
  - id: toy-calls
    loader: hf-chunked
    dataset_key: toy_calls
    source: { repo: org/toy-calls, split: test }
    chunked: { metadata_file: chunks.jsonl, parent_repo: org/toy-calls-full, parent_metadata_file: calls.jsonl }
"""
TEST_CASES = {"c0": "alpha beta gamma", "c1": "delta epsilon", "c2": "Uh", "c3": "twenty two apples in all"}
CALL = "good morning everyone and welcome"
TEST_CASES_SET, CALLS_SET = "org-toy_cases_test", "org-toy-calls__test"


def save_test_cases(paths: Paths, store: Store) -> None:
    test_cases = []
    references = {}
    for i, (sid, ref) in enumerate(TEST_CASES.items()):
        test_case = TestCase(f"toy_cases/{sid}")
        populate_metadata(test_case, {"audio_length_s": 2.0 + i}, ref, paths)
        test_cases.append(test_case)
        references[test_case.test_case_id] = ref
    store.test_cases.write("toy-cases", test_cases)
    store.references.write("toy-cases", references)


def save_calls(paths: Paths, store: Store) -> None:
    chunks = []
    for i in range(2):
        chunk = TestCase(f"toy_calls/k{i}", parent_id="call-1", chunk_index=i)
        populate_metadata(chunk, {"duration": 30.0}, CALL, paths)
        chunks.append(chunk)
    store.test_cases.write("toy-calls", chunks)
    store.references.write("toy-calls", {"toy_calls/call-1": CALL})


def runs_here(paths: Paths) -> list[tuple[str, str]]:
    """Which model has transcripts of which benchmark, benchmark by benchmark."""
    runs = Store(paths).hypotheses.runs(["lab/alpha", "lab/beta"], ["toy-cases", "toy-calls"])
    return [(run.system_id, run.benchmark_id) for run in runs]


def rows_of_test_cases(preds: dict[str, str]) -> list[dict]:
    return [{"audio_filepath": sid, "duration": 2.0, "time": 0.1, "text": TEST_CASES[sid], "pred_text": pred} for sid, pred in preds.items()]


def call_rows(first: str, second: str) -> list[dict]:
    return [
        {"audio_filepath": f"k{i}", "duration": 30.0, "time": 0.1, "text": CALL, "pred_text": text, "parent_id": "call-1", "chunk_index": i}
        for i, text in enumerate([first, second])
    ]


ALPHA_TEST_CASES = rows_of_test_cases({"c0": "alpha beta gama", "c1": "delta epsilon", "c3": "22 apples in all"})  # 1 error in 9 words
BETA_TEST_CASES = rows_of_test_cases({"c0": "alpha beta gamma", "c1": "", "c3": "22 apples"})  # 4 deletions in 9 words


class FakeBucket:
    """Stands in for the public bucket: what each model has there, by set."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.files: dict[tuple[str, str], dict] = {}
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    def put(self, system_id: str, dataset_id: str, rows: list[dict], score: dict | None = None, uploaded_at: str = "2026-11-02T14:14:38+00:00"):
        self.files[(system_id, dataset_id)] = {"rows": rows, "score": score, "uploaded_at": uploaded_at}

    def pull_manifests(self, paths, system_id, dataset_ids, api=None):
        self.asked.append((system_id, tuple(dataset_ids)))
        out = []
        for (owner, dataset_id), file in sorted(self.files.items()):
            if owner != system_id or dataset_id not in dataset_ids:
                continue
            folder = leaderboard.local_folder(paths, system_id)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"MODEL_{manifests.model_folder(system_id)}_DATASET_{dataset_id}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in file["rows"]), encoding="utf-8")
            if file["score"] is not None:
                manifests.score_file_for(path).write_text(json.dumps({**file["score"], "manifest_sha256": manifests.file_sha256(path)}))
            out.append(leaderboard.PublicFile(path, dataset_id, file["uploaded_at"]))
        return out


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A scratch project: two models and two small benchmarks, one of them scored per session."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "systems.yaml").write_text(SYSTEMS)
    (configs / "benchmarks.yaml").write_text(BENCHMARKS)
    third_party = REPO_ROOT / "third_party"
    paths = Paths(root=tmp_path, configs=configs, clone=third_party / "open_asr_leaderboard", urgent_clone=third_party / "urgent2024_analysis")
    monkeypatch.setattr(common, "get_paths", lambda: paths)
    return paths


@pytest.fixture
def bucket(project, monkeypatch):
    fake = FakeBucket(project)
    monkeypatch.setattr(leaderboard, "pull_manifests", fake.pull_manifests)
    return fake


@pytest.fixture
def lists(project, monkeypatch):
    """Stands in for the datasets: a pull saves a small list, and is remembered."""
    pulled = []

    def pull_benchmark(paths, store, cfg, benchmark_id, *, with_audio=False, **_):
        pulled.append((benchmark_id, with_audio))
        (save_test_cases if benchmark_id == "toy-cases" else save_calls)(paths, store)
        return len(store.test_cases.ids(benchmark_id))

    monkeypatch.setattr(benchmarks, "pull_benchmark", pull_benchmark)
    return pulled


@pytest.fixture
def fetched(project, bucket, lists):
    """Both models' transcripts of both benchmarks, fetched and imported."""
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES, score={"wer": 11.11, "ins": 0, "del": 0, "sub": 1})
    bucket.put("lab/alpha", CALLS_SET, call_rows("good morning everyone", "and welcome"))
    bucket.put("lab/beta", TEST_CASES_SET, BETA_TEST_CASES)
    bucket.put("lab/beta", CALLS_SET, call_rows("good morning", "and welcome"))
    assert invoke("fetch").exit_code == 0
    return project


def invoke(*args, input=None):
    return runner.invoke(main.app, list(args), input=input, catch_exceptions=False)


def blocks(output: str) -> list[list[str]]:
    """What a command printed, as its blocks of lines: a blank line parts one from the next."""
    return [block.splitlines() for block in output.strip().split("\n\n")]


def set_right(lines: list[str], heading: str) -> bool:
    """Whether a column of a table is set to the right: every cell ends where its heading does."""
    end = lines[0].index(heading) + len(heading)
    return all(len(row) >= end and row[end - 1] != " " and row[end : end + 1] in ("", " ") for row in lines[2:])


LEGEND = [["alpha", "lab/alpha"], ["beta", "lab/beta"]]  # each model's short name, and the id it is short for


# ── naming a model ───────────────────────────────────────────────────────────────────────


def test_an_unknown_model_is_one_line_that_says_what_can_be_given(project):
    result = invoke("score", "--system", "lab/gamma")
    assert result.exit_code == 1
    assert result.output.strip() == "Error: No system 'lab/gamma'. Configured: alpha, beta. A whole id works too: see configs/systems.yaml."
    assert "No benchmark 'toy-clops'. Configured: toy-cases, toy-calls" in invoke("score", "--benchmark", "toy-clops").output


@pytest.mark.parametrize("command", [["score", "--benchmark", "toy-cases"], ["ablate", "--benchmark", "toy-cases"], ["slice", "--by", "alphanumeric"]])
def test_a_model_can_be_chosen_by_the_short_name_the_tables_show(fetched, command):
    by_id, by_name = invoke(*command, "--system", "lab/beta"), invoke(*command, "--system", "beta")
    assert by_id.exit_code == 0 and by_name.exit_code == 0, by_name.output
    assert by_name.output == by_id.output and "lab/alpha" not in by_name.output  # that model, and no other


def test_a_short_name_is_taken_in_any_case_and_beside_a_whole_id(fetched):
    result = invoke("score", "--benchmark", "toy-cases", "--normalizer", "leaderboard", "--system", "BETA", "--system", "lab/alpha")
    assert result.exit_code == 0, result.output
    legend, scored = blocks(result.output)
    assert [line.split() for line in legend[2:]] == [["beta", "lab/beta"], ["alpha", "lab/alpha"]]  # in the order they were asked for
    assert [line.split()[0] for line in scored[3:]] == ["beta", "alpha"]


def test_fetch_takes_a_short_name_too(project, bucket, lists):
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES)
    bucket.put("lab/beta", TEST_CASES_SET, BETA_TEST_CASES)
    assert invoke("fetch", "--system", "beta", "--benchmark", "toy-cases").exit_code == 0
    assert [name for name, _ in bucket.asked] == ["lab/beta"]  # the bucket is asked by the model's id
    assert runs_here(project) == [("lab/beta", "toy-cases")]


# ── fetching ─────────────────────────────────────────────────────────────────────────────


def test_fetch_saves_the_lists_and_every_models_public_transcripts(project, bucket, lists):
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES, score={"wer": 11.11, "ins": 0, "del": 0, "sub": 1})
    bucket.put("lab/alpha", CALLS_SET, call_rows("good morning everyone", "and welcome"))
    bucket.put("lab/beta", TEST_CASES_SET, BETA_TEST_CASES)
    result = invoke("fetch")
    assert result.exit_code == 0, result.output
    assert lists == [("toy-cases", False), ("toy-calls", False)]  # the lists alone: no audio is fetched
    assert runs_here(project) == [("lab/alpha", "toy-cases"), ("lab/beta", "toy-cases"), ("lab/alpha", "toy-calls")]
    store = Store(project)
    # "Uh" is in no file: the leaderboard drops it before any model hears it. So it has no transcript.
    alpha = Run("lab/alpha", "toy-cases")
    assert list(store.hypotheses.read(alpha)) == ["toy_cases/c0", "toy_cases/c1", "toy_cases/c3"]
    # The leaderboard's own score of a file is kept beside ours, where it left one.
    assert store.published_scores.read(alpha) == {"wer": 11.11, "wer_sub": 1, "wer_del": 0, "wer_ins": 0}
    assert not store.published_scores.has(Run("lab/beta", "toy-cases"))
    # What the bucket does not have is said, and is not an error.
    assert "lab/beta on toy-calls: no public transcripts" in result.output
    assert "lab/alpha on toy-cases: 3 transcripts" in result.output and "lab/alpha on toy-calls: 2 transcripts" in result.output


def test_a_fetch_starts_over_and_warns_that_everything_stored_is_removed(project, bucket, lists):
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES)
    first = invoke("fetch")
    assert first.output.startswith("Warning: a fetch starts over.") and "Everything stored under artifacts/ is removed first" in first.output
    assert invoke("score", "--benchmark", "toy-cases").exit_code == 0
    assert (project.artifacts / "scores" / "ours").exists()  # something was worked out
    again = invoke("fetch")
    assert again.exit_code == 0 and again.output.startswith("Warning: a fetch starts over.")
    # Every list is read again, and what was worked out went with the rest.
    assert lists == [("toy-cases", False), ("toy-calls", False)] * 2
    assert not (project.artifacts / "scores" / "ours").exists()
    assert runs_here(project) == [("lab/alpha", "toy-cases")]


def test_fetch_can_be_limited_to_some_models_and_benchmarks(project, bucket, lists):
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES)
    bucket.put("lab/beta", TEST_CASES_SET, BETA_TEST_CASES)
    assert invoke("fetch").exit_code == 0 and len(runs_here(project)) == 2
    lists.clear()
    bucket.asked.clear()
    assert invoke("fetch", "--system", "lab/beta", "--benchmark", "toy-cases").exit_code == 0
    assert lists == [("toy-cases", False)]
    assert bucket.asked == [("lab/beta", (TEST_CASES_SET, "org-toy_old-name_test"))]  # the set's older name is looked for too
    # What is fetched is all there is afterwards: the other model and the other benchmark went with the rest.
    assert runs_here(project) == [("lab/beta", "toy-cases")] and not Store(project).test_cases.has("toy-calls")


def test_transcripts_under_a_sets_older_name_are_found_and_the_newest_file_stands(project, bucket, lists):
    bucket.put("lab/alpha", "org-toy_old-name_test", BETA_TEST_CASES, uploaded_at="2026-07-30T00:00:00+00:00")
    assert invoke("fetch", "--benchmark", "toy-cases").exit_code == 0
    store, alpha = Store(project), Run("lab/alpha", "toy-cases")
    assert store.hypotheses.read(alpha)["toy_cases/c3"] == "22 apples"
    # The leaderboard runs the model again, under the set's current name: that one is taken.
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES, uploaded_at="2026-11-02T00:00:00+00:00")
    assert invoke("fetch", "--benchmark", "toy-cases").exit_code == 0
    assert store.hypotheses.read(alpha)["toy_cases/c3"] == "22 apples in all" and runs_here(project) == [("lab/alpha", "toy-cases")]


def test_fetch_with_audio_also_downloads_the_audio(project, bucket, lists):
    result = invoke("fetch", "--benchmark", "toy-cases", "--audio")
    assert result.exit_code == 0 and lists == [("toy-cases", True)]
    assert "toy-cases: reading its test cases and downloading its audio ..." in result.output and "  4 test cases" in result.output


# ── the reproduction ─────────────────────────────────────────────────────────────────────


def reproduced(output: str) -> dict[tuple[str, str], list[str]]:
    """What `ba reproduce` shows for each model on each benchmark: the cells after the model's name."""
    rows = {}
    for lines in blocks(output):
        if len(lines) > 3 and lines[1].startswith("model  scored/test cases"):
            rows.update({(line.split()[0], lines[0]): line.split()[1:] for line in lines[3:]})
    return rows


def test_reproduce_puts_our_score_beside_the_leaderboards_own(fetched):
    result = invoke("reproduce")
    rows = reproduced(result.output)
    # alpha on toy-cases: 1 error in 9 words. Its score file says so, and the table's figure is another.
    assert rows[("alpha", "toy-cases")] == ["3/4", "11.11", "11.11", "20.00", "DIFFERS"]
    # alpha on toy-calls: no score file. 0 errors in 5 words, and the table says 12.50.
    assert rows[("alpha", "toy-calls")] == ["1/1", "0.00", "-", "12.50", "DIFFERS"]
    # beta on toy-cases: 4 deletions in 9 words is 44.44, and the table says 99.99.
    assert rows[("beta", "toy-cases")] == ["3/4", "44.44", "-", "99.99", "DIFFERS"]
    # beta on toy-calls: one word of five is missing, and nothing is published to compare with.
    assert rows[("beta", "toy-calls")] == ["1/1", "20.00", "-", "-", "nothing", "published"]
    assert result.exit_code == 1  # a figure that differs is a failure


def test_reproduce_scores_the_transcripts_as_written_whatever_is_saved(fetched):
    first = reproduced(invoke("reproduce").output)
    store, run = Store(fetched), Run("lab/alpha", "toy-cases")
    as_normalized = store.hypotheses.read(run, "leaderboard")
    assert as_normalized["toy_cases/c3"] == "22 apples in all"
    # As if the leaderboard's rules had changed since: the saved texts are no longer what they give.
    store.hypotheses.write(run, dict.fromkeys(as_normalized, ""), "leaderboard")
    store.references.write("toy-cases", dict.fromkeys(store.references.read("toy-cases", "leaderboard"), "other words"), "leaderboard")
    assert reproduced(invoke("reproduce").output) == first
    assert store.hypotheses.read(run, "leaderboard") == as_normalized


def test_reproduce_gives_each_benchmark_a_table_of_its_own_under_short_model_names(fetched):
    legend, test_cases, calls, *_ = blocks(invoke("reproduce").output)
    assert [line.split() for line in legend[2:]] == LEGEND
    assert (test_cases[0], calls[0]) == ("toy-cases", "toy-calls") and len(test_cases) == len(calls) == 1 + 2 + 2
    assert test_cases[1] == "model  scored/test cases  WER here  its score file  the table  verdict"
    assert all(set_right(test_cases[1:], heading) for heading in ("scored/test cases", "WER here", "its score file", "the table"))


def test_reproduce_passes_when_every_published_figure_is_matched(project, bucket, lists):
    bucket.put("lab/beta", TEST_CASES_SET, rows_of_test_cases({"c0": "alpha beta gamma", "c1": "delta epsilon", "c3": "22 apples in all"}), score={"wer": 0.0, "ins": 0, "del": 0, "sub": 0})
    invoke("fetch", "--system", "lab/beta", "--benchmark", "toy-cases")
    (project.configs / "systems.yaml").write_text(SYSTEMS.replace("toy-cases: 99.99", "toy-cases: 0.00"))
    result = invoke("reproduce")
    assert result.exit_code == 0, result.output
    assert reproduced(result.output)[("beta", "toy-cases")] == ["3/4", "0.00", "0.00", "0.00", "match"]
    assert "not fetched" in result.output  # the pairs with no transcripts here are named, and are no failure
    assert "2 of 2 published figures matched" in result.output  # its score file and the table


def test_reproduce_gives_each_models_mean_over_the_sets_the_table_shows(fetched):
    means = blocks(invoke("reproduce").output)[-1]
    assert means[0] == "Mean WER over the sets the table shows:" and means[1].split() == ["model", "here", "the", "table", "sets"]
    # alpha: (11.11 + 0.00) / 2 against the table's (20.00 + 12.50) / 2.
    assert means[3].split() == ["alpha", "5.56", "16.25", "2"]
    assert all(set_right(means[1:], heading) for heading in ("here", "the table", "sets"))


def test_a_score_file_with_the_same_figure_from_other_error_counts_is_told_apart(project, bucket, lists):
    # The leaderboard's rules have changed since some files were scored. The figure can
    # survive that to two decimals while the counts behind it do not.
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES, score={"wer": 11.11, "ins": 1, "del": 0, "sub": 0})  # one error, of another kind
    invoke("fetch", "--system", "lab/alpha", "--benchmark", "toy-cases")
    (project.configs / "systems.yaml").write_text(SYSTEMS.replace("toy-cases: 20.00", "toy-cases: 11.11"))
    result = invoke("reproduce")
    assert reproduced(result.output)[("alpha", "toy-cases")] == ["3/4", "11.11", "11.11", "11.11", "counts", "differ"]
    assert "2 of 2 published figures matched" in result.output
    counts = next(lines for lines in blocks(result.output) if lines[0] == "1 score file has the same WER from other error counts:")
    assert counts[1].split() == ["model", "benchmark", "counts", "ins", "del", "sub"]
    assert [line.split() for line in counts[3:]] == [["alpha", "toy-cases", "here", "0", "0", "1"], ["alpha", "toy-cases", "its", "score", "file", "1", "0", "0"]]
    assert result.exit_code == 0  # every published figure is reproduced


# ── scoring ──────────────────────────────────────────────────────────────────────────────


def test_score_prints_every_count_beside_the_number(fetched):
    result = invoke("score", "--system", "lab/beta", "--benchmark", "toy-cases", "--normalizer", "leaderboard")
    assert result.exit_code == 0, result.output
    legend, (title, names, rule, row) = blocks(result.output)
    assert [line.split() for line in legend[2:]] == [["beta", "lab/beta"]]
    # What is the same for every row is said once, above them: the benchmark, the metric, the one normalizer
    # that was asked for, and how many test cases the benchmark has.
    assert title == "toy-cases: WER, leaderboard normalizer, 4 test cases"
    assert names.split() == ["model", "scored", "ref", "units", "empty", "mean", "median", "overall"]
    cells = row.split()
    assert cells[:4] == ["beta", "3", "9", "1"]  # 3 of the 4 test cases are scored, in 9 words, and one transcript is empty
    assert cells[4:] == ["50.00", "50.00", "44.44"]  # mean, median, overall: test cases at 0%, 100% and 50%
    assert all(set_right([names, rule, row], column) for column in ("scored", "ref units", "empty", "mean", "median", "overall"))


def test_score_gives_each_benchmark_and_metric_a_table_of_every_model_and_normalizer(fetched):
    _, *tables = blocks(invoke("score").output)
    # A set scored per session counts sessions.
    assert [lines[0] for lines in tables] == ["toy-cases: WER, 4 test cases", "toy-calls: WER, 1 session"]
    # Six normalizers for each model: the normalizer is a column.
    assert all(lines[1].split()[:2] == ["model", "normalizer"] for lines in tables)
    normalizer_ids = ["verbatim", "case-punct", "whisper-stock", "leaderboard", "leaderboard_split", "own"]
    assert all([line.split()[:2] for line in lines[3:]] == [[model, rule] for model in ("alpha", "beta") for rule in normalizer_ids] for lines in tables)
    _, *two = blocks(invoke("score", "--metric", "wer", "--metric", "cer", "--normalizer", "verbatim", "--benchmark", "toy-cases").output)
    assert [lines[0] for lines in two] == ["toy-cases: WER, verbatim normalizer, 4 test cases", "toy-cases: CER, verbatim normalizer, 4 test cases"]
    assert all([line.split()[0] for line in lines[3:]] == ["alpha", "beta"] for lines in two)


def test_score_takes_a_rule_family_switched_off(fetched):
    off = invoke("score", "--normalizer", "own", "--without", "numerals", "--system", "lab/alpha", "--benchmark", "toy-cases").output
    # The variant's long name is said once, above the table, and not in a column that would make every row wide.
    title, names, *_ = blocks(off)[1]
    assert title == "toy-cases: WER, own-without-numerals normalizer, 4 test cases" and names.split()[:2] == ["model", "scored"]
    assert " 30.00" in off  # "twenty two" is now two words against "22"
    # The leaderboard's rules as families take a family off the same way.
    split = invoke("score", "--normalizer", "leaderboard_split", "--without", "numerals", "--system", "lab/alpha", "--benchmark", "toy-cases").output
    assert blocks(split)[1][0] == "toy-cases: WER, leaderboard_split-without-numerals normalizer, 4 test cases" and " 30.00" in split
    # A family of ours is no family of the leaderboard's rules.
    result = invoke("score", "--normalizer", "leaderboard_split", "--without", "repetitions")
    assert result.exit_code == 1 and "Unknown rule family: repetitions" in result.output
    assert invoke("score", "--normalizer", "own", "--without", "repetitions", "--benchmark", "toy-cases").exit_code == 0
    # No other normalizer has families.
    result = invoke("score", "--without", "numerals")
    assert result.exit_code == 1 and "--normalizer own" in result.output
    result = invoke("score", "--normalizer", "leaderboard", "--without", "numerals")
    assert result.exit_code == 1 and "--normalizer own" in result.output


def test_score_says_when_nothing_is_fetched(project):
    assert "Nothing fetched yet. Run: ba fetch" in invoke("score").output


def test_ablate_shows_what_each_family_forgives(fetched):
    result = invoke("ablate", "--benchmark", "toy-cases")
    assert result.exit_code == 0, result.output
    said, _, (heading, names, _, *rows) = blocks(result.output)
    assert said == ["What each rule family of the own normalizer is worth.", "First row: the score. Other rows: the change in points when that family is off."]
    assert heading == "toy-cases: overall WER" and names.split() == ["variant", "alpha", "beta"]
    assert rows[0].split()[-2:] == ["11.11", "44.44"]  # all families on: nothing here is for our four rules to forgive
    assert len(rows) == 18 and rows[-1].startswith("without repetitions")
    numerals = next(row for row in rows if row.startswith("without numerals"))
    # "twenty two" against "22": with numerals off, alpha's test case is charged for them.
    assert float(numerals.split()[-2]) > 0
    # A column's name and its figures are set to the right, so every line ends at the same place.
    assert {len(line) for line in (names, *rows)} == {len(rows[0])}


def test_ablate_takes_the_leaderboards_rules_as_families_too(fetched):
    result = invoke("ablate", "--normalizer", "leaderboard_split", "--benchmark", "toy-cases")
    assert result.exit_code == 0, result.output
    said, _, (heading, names, _, *rows) = blocks(result.output)
    assert said[0] == "What each rule family of the leaderboard_split normalizer is worth."
    assert rows[0].split()[-2:] == ["11.11", "44.44"]  # all families on: the leaderboard's own figure
    assert len(rows) == 14 and rows[-1].startswith("without initialisms")
    # Only the two normalizers that are made of families have a family table.
    result = invoke("ablate", "--normalizer", "leaderboard")
    assert result.exit_code == 1 and "own, leaderboard_split" in result.output


def test_ablate_leaves_a_model_out_of_the_table_of_a_benchmark_it_has_no_transcripts_of(project, bucket, lists):
    bucket.put("lab/alpha", TEST_CASES_SET, ALPHA_TEST_CASES)
    bucket.put("lab/beta", TEST_CASES_SET, BETA_TEST_CASES)
    bucket.put("lab/beta", CALLS_SET, call_rows("good morning", "and welcome"))
    assert invoke("fetch").exit_code == 0
    _, _, test_cases, calls = blocks(invoke("ablate").output)
    assert test_cases[1].split() == ["variant", "alpha", "beta"] and calls[1].split() == ["variant", "beta"]


def test_ablate_counts_each_family_once_and_reads_its_scores_after(fetched):
    first = invoke("ablate", "--benchmark", "toy-cases")
    assert first.exit_code == 0

    def written() -> dict[str, int]:
        return {str(file): file.stat().st_mtime_ns for file in (fetched.artifacts / "scores" / "ours").glob("*__toy-cases/*.parquet")}

    # A family switched off is a normalizer like any other, with a score file for each model.
    saved = written()
    assert len(saved) == 2 * 18 and len([name for name in saved if name.endswith("own-without-fillers.parquet")]) == 2
    # The same table, and another aggregation of it, with nothing counted again.
    assert invoke("ablate", "--benchmark", "toy-cases").output == first.output
    assert "mean WER" in invoke("ablate", "--benchmark", "toy-cases", "--agg", "mean").output
    assert written() == saved
    # Asked to, it counts every one of them again, and gives the same table.
    assert invoke("ablate", "--benchmark", "toy-cases", "--recompute").output == first.output
    assert all(written()[name] != stamp for name, stamp in saved.items())


def test_ablate_says_how_to_have_semantic_error_for_the_families(fetched):
    result = invoke("ablate", "--metric", "semantic")
    assert result.exit_code == 0, result.output
    assert "Semantic error is not computed for the rule families of own yet. Compute it with: ba semantic --normalizer own --families" in result.output


# ── semantic error ───────────────────────────────────────────────────────────────────────


class WordBag:
    """Stands in for the sentence encoder: a text is the bag of its words."""

    id = "test/word-bag"
    built = 0
    asked: list[str] = []
    WORDS = ["alpha", "beta", "gamma", "gama", "delta", "epsilon", "22", "apples", "in", "all", "twenty", "two", "uh"]

    def __init__(self, model: str = "x") -> None:
        type(self).built += 1

    def encode(self, texts):
        type(self).asked += texts
        return np.array([[text.lower().split().count(w) + 0.01 for w in self.WORDS] for text in texts])


@pytest.fixture
def encoder(monkeypatch):
    from ba.metrics import semantic

    WordBag.built, WordBag.asked = 0, []
    monkeypatch.setattr(semantic, "SentenceTransformerEmbedder", WordBag)
    return WordBag


def test_semantic_computes_the_metric_once_and_stores_it_beside_the_others(fetched, encoder):
    result = invoke("semantic", "--normalizer", "leaderboard")
    assert result.exit_code == 0, result.output
    assert "toy-cases, leaderboard normalizer: 2 models" in result.output
    # A set scored per session is passed over, and said to be.
    assert "toy-calls: left out" in result.output and "scored per session" in result.output
    assert encoder.asked.count("alpha beta gamma") == 1  # the two models share the reference
    _, shown, note = blocks(invoke("score", "--metric", "semantic", "--normalizer", "leaderboard", "--benchmark", "toy-cases").output)
    assert shown[0] == "toy-cases: semantic error, leaderboard normalizer, 4 test cases" and [line.split()[0] for line in shown[3:]] == ["alpha", "beta"]
    assert note == ["Semantic error is shown where it has been computed. Compute it with: ba semantic"]
    # Asked again, there is nothing to compute, and the encoder is not even loaded.
    encoder.built = 0
    again = invoke("semantic", "--normalizer", "leaderboard")
    assert "Nothing to compute" in again.output and encoder.built == 0
    assert "2 models" in invoke("semantic", "--normalizer", "leaderboard", "--recompute").output


def test_semantic_scores_the_rule_families_when_asked_and_the_family_table_shows_them(fetched, encoder):
    result = invoke("semantic", "--normalizer", "own", "--families")
    assert result.exit_code == 0, result.output
    # The normalizer itself, and each of its seventeen families switched off in turn.
    assert "toy-cases, own normalizer: 2 models" in result.output
    assert "toy-cases, own-without-numerals normalizer: 2 models" in result.output
    assert "Semantic error computed for 36 model and normalizer pairs" in result.output
    # A text that a family leaves as it was is embedded once for all of them.
    assert encoder.asked.count("alpha beta gamma") == 1
    table = invoke("ablate", "--metric", "semantic", "--benchmark", "toy-cases")
    assert table.exit_code == 0, table.output
    said, _, (heading, names, _, *rows) = blocks(table.output)
    assert heading == "toy-cases: overall semantic error" and names.split() == ["variant", "alpha", "beta"]
    assert len(rows) == 18 and rows[0].startswith("all families on")
    # A set scored per session has no semantic error, so it has no table.
    assert "toy-calls" not in invoke("ablate", "--metric", "semantic").output.split("First row")[1]
    # Asked again there is nothing to compute. And only a normalizer made of families has families.
    assert "Nothing to compute" in invoke("semantic", "--normalizer", "own", "--families").output
    refused = invoke("semantic", "--normalizer", "leaderboard", "--families")
    assert refused.exit_code == 1 and "own, leaderboard_split" in refused.output


# ── metadata and slices ──────────────────────────────────────────────────────────────────


def test_metadata_derives_again_what_is_worked_out_here_and_keeps_the_rest(fetched):
    store = Store(fetched)
    test_cases = store.test_cases.read("toy-cases")
    # As if the counting had changed since the fetch, and with a level measured on audio that is no longer on disk.
    test_cases[0].ref_words = 99
    test_cases[0].speaking_rate = 49.5
    test_cases[0].level = -20.5
    store.test_cases.write("toy-cases", test_cases)
    result = invoke("metadata", "--benchmark", "toy-cases")
    assert result.exit_code == 0 and result.output.strip() == "toy-cases: 4 test cases, 0 with audio on disk"
    first = store.test_cases.read("toy-cases")[0]
    # Counted again. What was measured on the audio, and what the source states, are as they were.
    assert (first.ref_words, first.level, first.duration) == (3, -20.5, 2.0)
    assert first.speaking_rate == 1.5  # the three words counted again, over its two seconds
    assert store.test_cases.read("toy-calls")[0].ref_words == 5  # the other benchmark was not asked for, and is untouched


def test_slice_breaks_a_score_down_with_its_counts(fetched):
    result = invoke("slice", "--by", "alphanumeric", "--system", "lab/beta")
    assert result.exit_code == 0, result.output
    said, legend, (title, names, rule, *rows), left_out = blocks(result.output)
    assert said == ["overall WER by alphanumeric, leaderboard normalizer"] and [line.split() for line in legend[2:]] == [["beta", "lab/beta"]]
    assert title == "toy-cases" and names == "model  alphanumeric  test cases  ref units  overall"
    # value, test cases, reference words, overall WER. The test case with a number: "22 apples" for "22 apples in all".
    assert [row.split() for row in rows] == [["beta", "false", "2", "5", "40.00"], ["beta", "true", "1", "4", "50.00"]]
    assert all(set_right([names, rule, *rows], column) for column in ("test cases", "ref units", "overall"))
    # A set scored per session has no test case to slice by. It is passed over, and said to be.
    assert left_out == ["Left out: toy-calls. A set scored per session has no test case to slice by."]


def test_slice_gives_a_number_a_row_for_each_value_or_cuts_it_into_the_bins_asked_for(fetched):
    result = invoke("slice", "--by", "duration", "--bins", "10", "--system", "beta", "--benchmark", "toy-cases")
    assert result.exit_code == 0, result.output
    _, _, (title, names, rule, *rows) = blocks(result.output)
    # Three test cases are scored, of three lengths: fewer values than bins, so each has a row of its own.
    assert [row.split() for row in rows] == [["beta", "2", "1", "3", "0.00"], ["beta", "3", "1", "2", "100.00"], ["beta", "5", "1", "4", "50.00"]]
    two = blocks(invoke("slice", "--by", "duration", "--bins", "2", "--system", "beta", "--benchmark", "toy-cases").output)[2]
    assert [row.split()[1:4] for row in two[3:]] == [["2.00", "to", "3.00"], ["3.00", "to", "5.00"]]


def test_slice_puts_every_model_of_a_benchmark_in_the_benchmarks_table(fetched):
    _, _, test_cases = blocks(invoke("slice", "--by", "alphanumeric", "--benchmark", "toy-cases").output)
    assert test_cases[0] == "toy-cases" and [row.split()[:2] for row in test_cases[3:]] == [["alpha", "false"], ["alpha", "true"], ["beta", "false"], ["beta", "true"]]


# ── clearing what was worked out here ────────────────────────────────────────────────────


def files(paths: Paths, folder: str) -> list[str]:
    return sorted(str(p.relative_to(paths.artifacts / folder)) for p in (paths.artifacts / folder).rglob("*") if p.is_file())


def fetched_files(paths: Paths) -> dict[str, bytes]:
    """What came from outside, file by file: the test cases, the texts as written and the leaderboard's own scores."""
    found = {}
    for pattern in ("test_cases/*.parquet", "transcriptions/*/*/clean.parquet", "scores/huggingface/*/*.parquet"):
        for file in sorted(paths.artifacts.glob(pattern)):
            found[str(file.relative_to(paths.artifacts))] = file.read_bytes()
    return found


@pytest.fixture
def worked(fetched, encoder):
    """A project with everything worked out: scores, semantic error and the rule families."""
    assert invoke("score", "--metric", "wer", "--metric", "cer").exit_code == 0
    assert invoke("semantic", "--normalizer", "leaderboard").exit_code == 0
    assert invoke("ablate").exit_code == 0
    return fetched


def test_clear_scores_removes_what_was_worked_out_and_never_what_was_fetched(worked):
    kept = fetched_files(worked)
    assert len(kept) == 2 + 2 + 4 + 1  # two lists, their references, four runs' transcripts, and one score file of the leaderboard's
    assert files(worked, "scores/ours")
    assert any(Path(name).stem != "clean" for name in files(worked, "transcriptions"))
    result = invoke("clear", "scores", input="y\n")
    assert result.exit_code == 0
    assert "Are you sure you want to remove scores? [y/N]" in result.output
    assert result.output.strip().endswith("Removed: scores")
    assert not (worked.artifacts / "scores" / "ours").exists()
    assert {Path(name).name for name in files(worked, "transcriptions")} == {"clean.parquet"}  # nothing a normalizer left
    # The test cases, the texts as written and the leaderboard's own score came from outside, and stay.
    assert fetched_files(worked) == kept
    # Everything can still be worked out from them.
    assert " 11.11" in invoke("score", "--system", "lab/alpha", "--benchmark", "toy-cases", "--normalizer", "leaderboard").output
    assert len(reproduced(invoke("reproduce").output)) == 4  # both models on both benchmarks


def test_clear_names_its_choices_when_given_something_else(worked):
    before = stored_and_downloaded_files(worked)
    result = invoke("clear", "transcripts")
    assert result.exit_code == 1 and "Choose one of: scores, leaderboard-scores, test-cases, audio, transcript-files" in result.output
    assert "Are you sure" not in result.output
    assert invoke("clear").exit_code == 2  # it has to be told what to remove
    assert stored_and_downloaded_files(worked) == before


def with_audio(paths: Paths) -> Store:
    """Audio on disk for the toy test cases, as `ba fetch --audio` leaves it."""
    store = Store(paths)
    test_cases = store.test_cases.read("toy-cases")
    for test_case in test_cases:
        test_case.audio_path = f"{test_case.test_case_id}.wav"
        (paths.audio / "toy_cases").mkdir(parents=True, exist_ok=True)
        (paths.audio / test_case.audio_path).write_bytes(b"RIFF" + bytes(96))
    store.test_cases.write("toy-cases", test_cases)
    return store


def stored_and_downloaded_files(paths: Paths) -> dict[Path, bytes]:
    return {
        file.relative_to(paths.root): file.read_bytes()
        for folder in (paths.artifacts, paths.data)
        for file in folder.rglob("*")
        if file.is_file()
    }


def test_clear_defaults_to_no_for_every_choice(worked):
    with_audio(worked)
    before = stored_and_downloaded_files(worked)
    for target in ("scores", "leaderboard-scores", "test-cases", "audio", "transcript-files"):
        result = invoke("clear", target, input="\n")
        assert result.exit_code == 1
        assert f"Are you sure you want to remove {target}? [y/N]" in result.output
        assert "Removed:" not in result.output
    assert stored_and_downloaded_files(worked) == before


@pytest.mark.parametrize(("target", "folder"), [("leaderboard-scores", "scores/huggingface"), ("test-cases", "test_cases")])
def test_clear_removes_only_the_named_artifact_directory(worked, target, folder):
    with_audio(worked)
    directory = worked.artifacts / folder
    relative = directory.relative_to(worked.root)
    before = stored_and_downloaded_files(worked)
    assert any(file.is_relative_to(relative) for file in before)

    result = invoke("clear", target, input="y\n")

    assert result.exit_code == 0
    assert f"Are you sure you want to remove {target}? [y/N]" in result.output
    assert result.output.strip().endswith(f"Removed: {target}")
    assert not directory.exists()
    assert stored_and_downloaded_files(worked) == {file: contents for file, contents in before.items() if not file.is_relative_to(relative)}


# ── clearing downloads ───────────────────────────────────────────────────────────────────


def test_clear_audio_removes_the_downloaded_audio_and_nothing_else(worked):
    store = with_audio(worked)
    kept = {folder: files(worked, folder) for folder in ("test_cases", "transcriptions", "scores")}
    downloads = sorted(str(p.relative_to(worked.leaderboard)) for p in worked.leaderboard.rglob("*") if p.is_file())
    result = invoke("clear", "audio", input="y\n")
    assert result.exit_code == 0 and result.output.strip().endswith("Removed: audio")
    assert not worked.audio.exists()
    assert {folder: files(worked, folder) for folder in kept} == kept
    assert sorted(str(p.relative_to(worked.leaderboard)) for p in worked.leaderboard.rglob("*") if p.is_file()) == downloads
    assert len(store.test_cases.read("toy-cases")) == 4  # the test cases are still on the list


def test_clear_transcript_files_removes_the_downloaded_files_and_keeps_the_transcripts_read_from_them(worked):
    hypotheses = files(worked, "transcriptions/hypotheses")
    assert len(list(worked.leaderboard.rglob("*.jsonl"))) == 4
    result = invoke("clear", "transcript-files", input="y\n")
    assert result.exit_code == 0 and not worked.leaderboard.exists()
    # The transcripts themselves are in the store: every score is still there to be had.
    assert files(worked, "transcriptions/hypotheses") == hypotheses
    assert " 11.11" in invoke("score", "--system", "lab/alpha", "--benchmark", "toy-cases", "--normalizer", "leaderboard").output


