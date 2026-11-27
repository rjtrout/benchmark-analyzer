"""The leaderboard's transcript files, and turning one into a model's transcripts of a benchmark.

A model's transcripts of a set are one JSONL file, a manifest: a line per test case with
``audio_filepath``, ``duration``, ``time``, ``text`` (the reference) and ``pred_text`` (the
transcript). The file is named ``MODEL_<model>_DATASET_<repo>_<config>_<split>.jsonl``, and
the leaderboard's scorer leaves its result beside it as ``<same name>.score.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run

SCORE_SUFFIX = ".score.json"


def model_folder(system_id: str) -> str:
    """How the leaderboard names a model in file and folder names."""
    return system_id.replace("/", "-")


def parse_manifest_filename(name: str) -> tuple[str, str]:
    """``(model folder, dataset id)`` from a manifest's file name: ``MODEL_<model>_DATASET_<dataset>.jsonl``."""
    stem = Path(name).name.removesuffix(".jsonl").removeprefix("MODEL_")
    model, dataset = stem.split("_DATASET_", 1)
    return model, dataset


def read_manifest(path: Path | str) -> list[dict[str, Any]]:
    """A manifest's rows: one JSON object to a line."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_file_for(manifest_path: Path | str) -> Path:
    return Path(str(manifest_path).removesuffix(".jsonl") + SCORE_SUFFIX)


def published_score(manifest_path: Path | str) -> dict[str, Any] | None:
    """The leaderboard's own score of this very file, if it left one beside it.

    A score file names the checksum of the manifest it was computed from. One written for
    another version of the file says nothing about this one, and is not used.
    """
    path = score_file_for(manifest_path)
    if not path.is_file():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("manifest_sha256") != file_sha256(manifest_path):
        return None
    return {"wer": stored.get("wer"), "wer_sub": stored.get("sub"), "wer_del": stored.get("del"), "wer_ins": stored.get("ins")}


def file_name_of(source_id: str) -> str:
    """The file name at the end of a test case ID, which is all most runners write of it."""
    return os.path.basename(str(source_id).strip())


def _source_id(test_case: TestCase) -> str:
    """A test case's id in its dataset: its id here, without the dataset in front."""
    return test_case.test_case_id.split("/", 1)[1]


def _join_by_session(rows: list[dict[str, Any]], test_cases: list[TestCase]) -> dict[str, dict[str, Any]]:
    """Each row's test case, on a set scored per session: the chunk at the row's place in its session.

    Every runner writes the session and the place, whatever it writes for the test case's name.
    """
    test_case_at = {}  # (session, place in it) -> test case id
    for test_case in test_cases:
        test_case_at[(test_case.parent_id, test_case.chunk_index)] = test_case.test_case_id

    row_of = {}
    for row in rows:
        test_case_id = test_case_at.get((row.get("parent_id"), row.get("chunk_index")))
        if test_case_id is not None and test_case_id not in row_of:
            row_of[test_case_id] = row
    return row_of


def _join_by_name(
    rows: list[dict[str, Any]],
    test_cases: list[TestCase],
    references: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Each row's test case, on any other set. ``references`` are the test cases' references as written.

    A row that gives a whole test case ID goes to that test case. Most runners write only the
    file name at its end, and in SPGISpeech thousands of test cases share one
    (``<call>/12.wav``), so among the test cases with that file name the row goes to the one
    with its reference. Two test cases with the same file name and the same reference are
    taken in the order of the list: a swap between them would change no score.
    """
    test_case_with_id = {}  # the test case's whole id in its dataset -> test case id
    test_cases_named: dict[str, list[str]] = {}  # the file name at the end of an id -> the test cases that share it
    for test_case in test_cases:
        source_id = _source_id(test_case)
        test_case_with_id[source_id] = test_case.test_case_id
        name = file_name_of(source_id)
        if name not in test_cases_named:
            test_cases_named[name] = []
        test_cases_named[name].append(test_case.test_case_id)

    row_of = {}
    for row in rows:
        written = str(row.get("audio_filepath", "")).strip()
        test_case_id = test_case_with_id.get(written)
        if test_case_id is None or test_case_id in row_of:
            # Not a whole id, or one that is taken: go by the file name.
            candidates = test_cases_named.get(file_name_of(written), [])
            if len(candidates) > 1:
                # Several test cases share the name. The row's own reference says which it is.
                reference = str(row.get("text") or "").strip()
                candidates = [c for c in candidates if references[c].strip() == reference]
            free = [c for c in candidates if c not in row_of]
            if free:
                test_case_id = free[0]
            else:
                test_case_id = None
        if test_case_id is not None:
            row_of[test_case_id] = row
    return row_of


def _session_transcripts(test_cases: list[TestCase], row_of: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Each session's transcript: its chunks' transcripts joined in order, as the leaderboard joins them.

    A chunk the model wrote nothing for adds nothing. A session with no row for any chunk is left out.
    """
    chunks_of: dict[str, list[TestCase]] = {}  # session -> its chunks that have a row
    for test_case in test_cases:
        if test_case.test_case_id in row_of:
            chunks_of.setdefault(test_case.unit_id, []).append(test_case)

    transcripts = {}
    for session_id, chunks in chunks_of.items():
        pieces = []
        for chunk in sorted(chunks, key=lambda chunk: chunk.chunk_index or 0):
            piece = (row_of[chunk.test_case_id].get("pred_text") or "").strip()
            if piece:
                pieces.append(piece)
        transcripts[session_id] = " ".join(pieces)
    return transcripts


def import_manifest(store: Store, manifest_path: Path | str, *, system_id: str, benchmark_id: str) -> int:
    """Turn a manifest into a model's transcripts of a benchmark. Returns how many of its rows found their test case.

    Each row is joined to its test case. A test case with no row has no transcript: the
    leaderboard's runner drops a test case whose normalized reference is empty before the
    model runs, and such a test case is scored for no model. On a set scored per session
    the transcripts are saved by session, as they are scored. The leaderboard's own score of
    the file, where it left one, is kept beside ours.
    """
    manifest_path = Path(manifest_path)
    rows = read_manifest(manifest_path)
    test_cases = store.test_cases.read(benchmark_id)
    if store.test_cases.scored_per_session(benchmark_id):
        row_of = _join_by_session(rows, test_cases)
        hypotheses = _session_transcripts(test_cases, row_of)
    else:
        row_of = _join_by_name(rows, test_cases, store.references.read(benchmark_id))
        hypotheses = {}
        for test_case in test_cases:
            if test_case.test_case_id in row_of:
                hypotheses[test_case.test_case_id] = row_of[test_case.test_case_id].get("pred_text") or ""
    run = Run(system_id, benchmark_id)
    store.hypotheses.write(run, hypotheses)

    score = published_score(manifest_path)
    if score is not None:
        store.published_scores.write(run, score)
    return len(row_of)
