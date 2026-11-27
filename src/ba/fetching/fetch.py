"""What ``ba fetch`` does: each benchmark's test case list, then each model's public transcripts of it."""

from __future__ import annotations

from collections.abc import Callable

from ba.config import Config
from ba.fetching import benchmarks, leaderboard, manifests
from ba.fetching.leaderboard import PublicFile
from ba.paths import Paths
from ba.store.store import Store
from ba.utils.display_utils import plural


def _newest(files: list[PublicFile], dataset_ids: list[str]) -> PublicFile | None:
    """A model's newest transcript file of a benchmark, or None if it has none.

    A set that was renamed can have transcripts under both of its names. The newest file
    is the model's result.
    """
    newest = None
    for file in files:
        if file.dataset_id not in dataset_ids:
            continue
        if newest is None or (file.uploaded_at or "") >= (newest.uploaded_at or ""):
            newest = file
    return newest


def fetch(
    paths: Paths,
    cfg: Config,
    store: Store,
    system_ids: list[str],
    benchmark_ids: list[str],
    audio: bool = False,
    say: Callable[[str], None] = print,
) -> None:
    """Fetch each benchmark's test cases and references, then each model's public transcripts of it.

    A fetch starts over. Everything stored is removed first, and everything is read again
    with no check against what was here. ``say`` is told each step as it is done.
    """
    say("Warning: a fetch starts over. Everything stored under artifacts/ is removed first: the test cases, the texts and every score.")
    store.remove_everything()

    for benchmark_id in benchmark_ids:
        if audio:
            say(f"{benchmark_id}: reading its test cases and downloading its audio ...")
        else:
            say(f"{benchmark_id}: reading its test cases ...")
        test_cases = benchmarks.pull_benchmark(paths, store, cfg, benchmark_id, with_audio=audio)
        say(f"  {plural(test_cases, 'test case')}")

    every_dataset_id = []  # the names the chosen benchmarks' transcript files go by
    for benchmark_id in benchmark_ids:
        every_dataset_id.extend(cfg.benchmark(benchmark_id).manifest_dataset_ids)

    for system_id in system_ids:
        files = leaderboard.pull_manifests(paths, system_id, every_dataset_id)
        for benchmark_id in benchmark_ids:
            file = _newest(files, cfg.benchmark(benchmark_id).manifest_dataset_ids)
            if file is None:
                say(f"{system_id} on {benchmark_id}: no public transcripts")
                continue
            transcripts = manifests.import_manifest(store, file.path, system_id=system_id, benchmark_id=benchmark_id)
            say(f"{system_id} on {benchmark_id}: {plural(transcripts, 'transcript')}")
