"""The store: everything kept under ``artifacts/``, as Parquet files, a table at a time.

    store.test_cases          test_cases/<benchmark>.parquet
        a benchmark's test cases, in order, each with its metadata
    store.references          transcriptions/references/<benchmark>/clean.parquet and <normalizer>.parquet
        the benchmark's references: as written, and as each normalizer leaves them
    store.hypotheses          transcriptions/hypotheses/<model>__<benchmark>/clean.parquet and <normalizer>.parquet
        a model's transcripts of the benchmark: as written, and as each normalizer leaves them
    store.scores              scores/ours/<model>__<benchmark>/<normalizer>.parquet
        the metrics worked out here, a row per transcript: a metric's columns are blank until it is computed
    store.published_scores    scores/huggingface/<model>__<benchmark>/leaderboard.parquet
        the leaderboard's own score of the model's transcript file

Each table is made from the ones above it. A fetch writes the test cases, the references
and the transcripts as written, and the leaderboard's own scores. A normalizer's texts are
made from the texts as written, and a metric's scores from a normalizer's two texts.

Every table has the same three verbs: ``has`` says whether something is kept, ``read`` gives
it, and ``write`` keeps it. Nothing here works anything out. What is not kept yet is made in
``ba.actions``, which asks ``has`` first and keeps what it makes.

The store is a snapshot. Nothing in it says what it was made under: after a change to a
normalizer, ``ba clear scores`` removes what was worked out here, and it is worked out again.

A model's transcripts of a benchmark are a ``Run``. A rule family switched off is a
normalizer like any other, with files of its own: ``own-without-numerals``. Every text and every score is kept under
the id of what is scored: ``test_case_id``. That is a test case's id, or a session's on a set
that is scored per session, where the test cases are chunks of sessions and a session has
one reference and one transcript.
"""

from __future__ import annotations

import shutil

from ba.paths import Paths
from ba.store.scores import PublishedScoreTable, ScoreTable
from ba.store.test_cases import TestCaseTable
from ba.store.transcriptions import CLEAN, HypothesisTable, ReferenceTable

# What ``ba clear`` can remove from the store and download cache.
CLEARABLE = ("scores", "leaderboard-scores", "test-cases", "audio", "transcript-files", "all")


class Store:
    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.root = paths.artifacts
        # The tables, in the order each is made from the ones before it.
        self.test_cases = TestCaseTable(self.root / "test_cases")
        self.references = ReferenceTable(self.root / "transcriptions" / "references")
        self.hypotheses = HypothesisTable(self.root / "transcriptions" / "hypotheses")
        self.scores = ScoreTable(self.root / "scores" / "ours")
        self.published_scores = PublishedScoreTable(self.root / "scores" / "huggingface")

    def remove_everything(self) -> None:
        """Remove everything stored here: the test cases, the texts and every score. A fetch starts with this."""
        shutil.rmtree(self.root, ignore_errors=True)

    def clear(self, kind: str) -> None:
        """Remove one category of stored artifacts or downloads, or every one of them with ``all``."""
        if kind not in CLEARABLE:
            raise ValueError(f"Choose one of: {', '.join(CLEARABLE)}")
        if kind == "scores" or kind == "all":
            shutil.rmtree(self.scores.folder, ignore_errors=True)
            # In a folder of texts, clean.parquet is the text as written. The rest is a normalizer's.
            for texts in (self.references, self.hypotheses):
                for file in texts.folder.glob("*/*.parquet"):
                    if file.stem != CLEAN:
                        file.unlink()
        if kind == "leaderboard-scores" or kind == "all":
            shutil.rmtree(self.published_scores.folder, ignore_errors=True)
        if kind == "test-cases" or kind == "all":
            shutil.rmtree(self.test_cases.folder, ignore_errors=True)
        if kind == "audio" or kind == "all":
            shutil.rmtree(self.paths.audio, ignore_errors=True)
        if kind == "transcript-files" or kind == "all":
            shutil.rmtree(self.paths.leaderboard, ignore_errors=True)
