"""The Open ASR Leaderboard's public bucket: where every open model's transcripts are.

One folder per model, one manifest per set, most with the score the leaderboard's own
scorer gave them beside it. It is public: nothing here uses an account or a token.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ba.fetching.manifests import SCORE_SUFFIX, model_folder, parse_manifest_filename
from ba.paths import Paths

PUBLIC_BUCKET = "hf-audio/asr_leaderboard_h200"


@dataclass(frozen=True)
class PublicFile:
    """A manifest in the local cache, and what the bucket says about it."""

    path: Path
    dataset_id: str  # the part of its name after _DATASET_
    uploaded_at: str | None  # when the leaderboard put it there


def local_folder(paths: Paths, system_id: str) -> Path:
    return paths.leaderboard / PUBLIC_BUCKET.replace("/", "-") / model_folder(system_id)


def _uploaded(entry: Any) -> datetime | None:
    """When the bucket says a file was put there, with its time zone. None if it does not say."""
    uploaded = getattr(entry, "uploaded_at", None)
    if uploaded is None:
        return None
    if uploaded.tzinfo is None:
        uploaded = uploaded.replace(tzinfo=UTC)
    return uploaded


def _is_manifest(name: str) -> bool:
    """A transcript file that lies in the model's own folder, and not in a folder below it."""
    return "/" not in name and name.startswith("MODEL_") and name.endswith(".jsonl")


def pull_manifests(paths: Paths, system_id: str, dataset_ids: list[str], api: Any = None) -> list[PublicFile]:
    """Download a model's manifests for these sets, and the score files beside them.

    Every file is downloaded, whatever is here already. Files go to the git-ignored cache.
    A model with no transcripts for a set gives nothing for it.
    """
    if api is None:
        from huggingface_hub import HfApi  # slow to import, and only a fetch needs it

        api = HfApi()

    # What the model's folder in the bucket holds: each file's name, and what the bucket says of it.
    prefix = model_folder(system_id) + "/"
    in_bucket = {}
    for entry in api.list_bucket_tree(PUBLIC_BUCKET, prefix=prefix, recursive=True, token=False):
        is_a_file = getattr(entry, "size", None) is not None  # a folder has no size
        if is_a_file:
            name = entry.path[len(prefix) :]
            in_bucket[name] = entry

    folder = local_folder(paths, system_id)
    manifests = []
    to_download = []
    for name in sorted(in_bucket):
        if not _is_manifest(name):
            continue
        _, dataset_id = parse_manifest_filename(name)
        if dataset_id not in dataset_ids:
            continue
        uploaded = _uploaded(in_bucket[name])
        uploaded_at = None
        if uploaded is not None:
            uploaded_at = uploaded.isoformat(timespec="seconds")
        manifests.append(PublicFile(folder / name, dataset_id, uploaded_at))

        # The transcript file, and the leaderboard's score file beside it where there is one.
        score_name = name.removesuffix(".jsonl") + SCORE_SUFFIX
        for wanted in (name, score_name):
            if wanted in in_bucket:
                to_download.append((in_bucket[wanted], str(folder / wanted)))

    if to_download:
        folder.mkdir(parents=True, exist_ok=True)
        api.download_bucket_files(PUBLIC_BUCKET, to_download, token=False)
    return manifests
