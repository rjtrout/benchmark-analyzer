"""Shared fixtures. No test here touches the network."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ba.paths import Paths
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    """A scratch project: artifacts and data under tmp, configs and the two clones from the repo."""
    return Paths(
        root=tmp_path,
        configs=REPO_ROOT / "configs",
        clone=REPO_ROOT / "third_party" / "open_asr_leaderboard",
        urgent_clone=REPO_ROOT / "third_party" / "urgent2024_analysis",
    )


def tone_bytes(seconds: float = 0.5, sr: int = 16000, fmt: str = "FLAC", freq: float = 440.0) -> bytes:
    """A short sine tone, encoded. Stands in for a test case's audio."""
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, 0.3 * np.sin(2 * np.pi * freq * t), sr, format=fmt)
    return buf.getvalue()


def leaderboards_score(paths: Paths, rows: list[dict]) -> dict:
    """The leaderboard's own score of a manifest's rows, from its own scorer: the check on ours."""
    from ba.normalize import leaderboard

    leaderboard.ensure_importable(paths)
    from normalizer.eval_utils import score_manifest

    return score_manifest(rows, num_workers=1)


# One transcript's row for one metric, as scoring hands it to an aggregation.
def row(test_case_id, n_ref, n_sub=0, n_del=0, n_ins=0, empty=False, rate=None):
    return dict(test_case_id=test_case_id, n_ref=n_ref, n_sub=n_sub, n_del=n_del, n_ins=n_ins, empty=empty, rate=rate)


TOY_RUN = Run("fake/sys", "toy-bench")


# A benchmark of three test cases with one model's transcripts of it. The third test case, "uh", is one the
# leaderboard never gives a model, so no model has a transcript of it: ``skipped`` leaves its row out.
def make_run(paths, *, statuses=("ok", "ok", "skipped"), system_id="fake/sys"):
    store = Store(paths)
    references = {"toy/c0": "alpha beta gamma", "toy/c1": "delta epsilon", "toy/c2": "uh"}
    test_cases = []
    for test_case_id, reference in references.items():
        test_cases.append(TestCase(test_case_id, duration=1.0, ref_words=len(reference.split()), alphanumeric=False))
    store.test_cases.write("toy-bench", test_cases)
    store.references.write("toy-bench", references)
    texts = {"ok": "alpha beta gama", "empty": ""}
    hypotheses = {}
    for test_case, status in zip(test_cases, statuses, strict=True):
        if status != "skipped":
            hypotheses[test_case.test_case_id] = texts[status]
    store.hypotheses.write(Run(system_id, "toy-bench"), hypotheses)
    return store
