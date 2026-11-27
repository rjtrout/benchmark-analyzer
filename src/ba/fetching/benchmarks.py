"""Benchmarks: reading a dataset's test cases and saving the list every model is scored on.

A benchmark is its test cases, saved in the dataset's order. It holds every test case of
the source, including those whose reference a normalizer would empty: the leaderboard drops
those as it goes, which is the reason to keep the list instead.

A pull reads the test cases' IDs and references, and fills in their metadata fields. The
audio is fetched only when asked for: the fields that come from the audio are filled then.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ba.config import Config
from ba.fetching.loaders import ID_COLUMNS, TEXT_COLUMNS, ParquetSource, resolve_hf
from ba.metadata.populate import populate_metadata, populate_speaking_rate
from ba.paths import Paths
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.utils.audio_utils import probe


class Hub:
    """How the chunked loader reaches the Hugging Face Hub, with no account. Tests pass a stand-in.

    The Hub's library is imported where it is used: it is slow to import, and only a fetch needs it.
    """

    def fetch_file(self, repo: str, filename: str, revision: str | None = None) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, token=False))

    def fetch_audio(self, repo: str, audio_dir: str, revision: str | None = None) -> Path:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{audio_dir}/*"],
            token=False,
        )
        return Path(root) / audio_dir

    def revision(self, repo: str) -> str:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo, token=False).sha


def _file_stem(source_id: str) -> str:
    """A file name for a test case's audio, made from its id.

    Any character a file system may not take becomes an underscore. Two ids could then give
    one name, so a name that was changed also gets a few characters of the id's checksum.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", source_id)
    if safe != source_id:
        safe += "-" + hashlib.sha256(source_id.encode()).hexdigest()[:8]
    return safe


def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    """The value of the first of these columns that the row has a value for, or None."""
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return None


def _audio_column(row: dict[str, Any]) -> dict[str, Any] | None:
    """A row's audio as the Hub stores it, with its bytes and its file's path. None when the audio was not read."""
    for value in row.values():
        if isinstance(value, dict) and value.get("bytes"):
            return value
    return None


def pull_benchmark(
    paths: Paths,
    store: Store,
    cfg: Config,
    benchmark_id: str,
    *,
    source: ParquetSource | None = None,
    with_audio: bool = False,
    hub: Hub | None = None,
) -> int:
    """Read a benchmark's test cases and references from its dataset as it is now, and save them.

    Returns how many test cases there are.
    """
    bench = cfg.benchmark(benchmark_id)
    if bench.loader == "hf-chunked":
        return _pull_chunked(paths, store, bench, with_audio, hub or Hub())
    if source is None:
        source = resolve_hf(bench.source)
    dataset_key = bench.dataset_key

    test_cases = []
    references = {}  # each test case's reference as written
    for index, row in enumerate(source.rows(with_audio)):
        source_id = _first(row, ID_COLUMNS)
        if not source_id:
            source_id = f"row-{index:07d}"
        source_id = str(source_id)
        test_case = TestCase(test_case_id=f"{dataset_key}/{source_id}")
        reference = str(_first(row, TEXT_COLUMNS) or "")
        audio = _audio_column(row)
        if audio is not None:
            data = audio["bytes"]
            suffix = Path(audio.get("path") or "").suffix.lower()
            if not suffix:
                _, container = probe(data)
                suffix = f".{container.lower()}"
            relative = Path(dataset_key) / f"{_file_stem(source_id)}{suffix}"
            target = paths.audio / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            test_case.audio_path = str(relative)
        populate_metadata(test_case, row, reference, paths)
        test_cases.append(test_case)
        references[test_case.test_case_id] = reference
    populate_speaking_rate(test_cases)
    store.test_cases.write(benchmark_id, test_cases)
    store.references.write(benchmark_id, references)
    return len(test_cases)


# ── chunked sets: audio in chunks, one reference per session ─────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """A file with one JSON object to a line."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pull_chunked(paths: Paths, store: Store, bench: Any, with_audio: bool, hub: Hub) -> int:
    """Earnings22's layout: an index of chunks, their audio files, and the session texts in a second repo.

    Each chunk becomes a test case that knows its session and its place in it, and each
    session has its one reference. A session is what is scored, as the leaderboard scores it.
    """
    spec = bench.chunked
    repo = bench.source.repo
    dataset_key = bench.dataset_key
    revision = hub.revision(repo)  # the index and the audio are read at one revision of the repo

    chunks = _read_jsonl(hub.fetch_file(repo, spec.metadata_file, revision))
    reference_of = {}  # session id -> its reference
    for session in _read_jsonl(hub.fetch_file(spec.parent_repo, spec.parent_metadata_file)):
        text = _first(session, (spec.text_column, *TEXT_COLUMNS))
        reference_of[str(session["id"])] = str(text or "")

    audio_folder = None
    if with_audio:
        audio_folder = hub.fetch_audio(repo, spec.audio_dir, revision)

    test_cases = []
    references = {}  # each session's reference as written, by the session's id here
    for chunk in chunks:
        session_id = str(chunk["parent_id"])
        test_case = TestCase(
            test_case_id=f"{dataset_key}/{chunk['id']}",
            parent_id=session_id,
            chunk_index=int(chunk["chunk_index"]),
        )
        if audio_folder is not None:
            relative = Path(dataset_key) / chunk["file_name"]
            target = paths.audio / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio_folder / chunk["file_name"], target)
            test_case.audio_path = str(relative)
        populate_metadata(test_case, chunk, reference_of[session_id], paths)
        test_cases.append(test_case)
        references[test_case.unit_id] = reference_of[session_id]
    populate_speaking_rate(test_cases)
    store.test_cases.write(bench.id, test_cases)
    store.references.write(bench.id, references)
    return len(test_cases)
