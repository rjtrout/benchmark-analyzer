"""The reproduction, on the real transcripts: ours against what the leaderboard publishes.

These read the test cases, transcripts and scores stored under ``artifacts/`` (``ba fetch``
and ``ba score`` put them there: public data, text only) and the published figures: the
leaderboard's own score of each transcript file, kept beside ours, and its table as read
into ``configs/systems.yaml``. Nothing is downloaded, and nothing is written: a score that
is not saved is an error that says how to save it, and is not worked out here.

    uv run pytest -m golden
"""

from __future__ import annotations

import pytest

from ba.actions.reproduce import reproduction
from ba.actions.scoring import score
from ba.cli.reproduce import table_means
from ba.config import load_config
from ba.normalize.registry import normalizers
from ba.paths import Paths
from ba.store.store import Store
from tests.conftest import REPO_ROOT

REPO = Paths(root=REPO_ROOT)
FETCHED = any((REPO.artifacts / "transcriptions" / "hypotheses").glob("*/clean.parquet"))
MUSCI, NEMOTRON = "Musci-research/Musci-ASR-qwen-1019-hf", "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
READ = ("leaderboard", "leaderboard_split")  # the normalizers whose saved word errors these tests read

pytestmark = [pytest.mark.golden, pytest.mark.skipif(not FETCHED, reason="No transcripts under artifacts/. Run: ba fetch")]


@pytest.fixture(scope="module", autouse=True)
def every_score_read_here_is_saved():
    """Scoring keeps what it works out, so a score that is not saved would be written into the store. These tests only read."""
    store, cfg = Store(REPO), load_config(REPO)
    runs = store.hypotheses.runs([system.id for system in cfg.systems()], [benchmark.id for benchmark in cfg.benchmarks()])
    missing = [f"{run.run_id} under {normalizer_id}" for run in runs for normalizer_id in READ if not store.scores.has(run, normalizer_id, "wer")]
    assert not missing, (
        f"Not every score is saved: {len(missing)} missing, such as {missing[0]}. "
        "Save them with: uv run ba score --normalizer leaderboard --normalizer leaderboard_split"
    )


@pytest.fixture(scope="module")
def rows():
    return reproduction(Store(REPO), load_config(REPO), normalizers(REPO).get("leaderboard"))


def test_every_model_has_transcripts_that_cover_every_benchmark(rows):
    assert len(rows) == 5 * 8
    assert {row["verdict"] for row in rows} <= {"match", "counts differ", "DIFFERS"}  # every pair has a published figure
    # The leaderboard scores every test case but those whose reference its rules empty: 90 in AMI, 11 in GigaSpeech.
    left_out = {row["benchmark"]: row["test_cases"] - row["scored"] for row in rows}
    assert {b: n for b, n in left_out.items() if n} == {"ami-cleaned": 90, "gigaspeech-cleaned": 11}


def test_every_score_file_the_leaderboard_published_is_reproduced(rows):
    with_file = [row for row in rows if row["score_file"] is not None]
    assert len(with_file) == 3 * 8  # three of the five models have them
    for row in with_file:
        assert row["score_file"]["wer"] == round(row["ours"], 2), (row["system"], row["benchmark"])
    other = {(row["system"], row["benchmark"]): (row["counts"], {k: row["score_file"][k] for k in ("ins", "del", "sub")}) for row in with_file if not row["counts_match"]}
    # One file reaches the same WER from two more errors in 968,595 words. It was scored on
    # 2 November 2026, the day before the leaderboard changed its normalizer (its commit
    # f58d58a, "Preserve bracketed speech"). The leaderboard's scorer of today gives our counts.
    assert other == {(MUSCI, "spgispeech"): ({"ins": 4030, "del": 3744, "sub": 10223}, {"ins": 4030, "del": 3745, "sub": 10224})}


def test_the_tables_figures_are_reproduced_for_four_models_and_are_near_for_the_fifth(rows):
    shown = [row for row in rows if row["table"] is not None]
    assert len(shown) == 5 * 8  # URGENT24 Clean too: a column of the table that its page leaves switched off
    off = {(row["system"], row["benchmark"]): round(round(row["ours"], 2) - row["table"], 2) for row in shown if round(row["ours"], 2) != row["table"]}
    # NVIDIA's public transcripts are not the run the table shows: seven of its eight figures
    # are off by a few hundredths of a point, in both directions. The leaderboard's own scorer
    # gives ours for these files. Its figure for URGENT24 Clean is the one that matches.
    assert {system for system, _ in off} == {NEMOTRON} and len(off) == 7
    assert (NEMOTRON, "urgent-clean") not in off
    assert max(abs(delta) for delta in off.values()) <= 0.05
    means = {mean["system"]: mean for mean in table_means(rows)}
    assert all(mean["sets"] == 8 for mean in means.values())
    assert all(round(mean["ours"], 2) == round(mean["table"], 2) for system, mean in means.items() if system != NEMOTRON)
    assert abs(means[NEMOTRON]["ours"] - means[NEMOTRON]["table"]) < 0.01


def test_leaderboard_split_with_every_family_on_scores_every_test_case_as_the_leaderboard_does():
    """leaderboard_split is the leaderboard's rules as families. All on, the two must give the same counts."""
    store, cfg = Store(REPO), load_config(REPO)
    registry = normalizers(REPO)
    runs = store.hypotheses.runs([s.id for s in cfg.systems()], [b.id for b in cfg.benchmarks()])
    assert len(runs) == 40
    test_cases = 0
    for run in runs:
        board = score(store, run, registry.get("leaderboard"))
        split = score(store, run, registry.get("leaderboard_split"))
        counts = lambda rows: [(r["test_case_id"], r["n_ref"], r["n_sub"], r["n_del"], r["n_ins"]) for r in rows]  # noqa: E731
        assert counts(split) == counts(board), run.run_id
        test_cases += len(board)
    assert test_cases > 300_000
