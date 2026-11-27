"""``ba metadata``: derive again the metadata fields that are worked out here."""

from __future__ import annotations

import typer

from ba.cli.common import BenchmarkOption, guard, picked, setup
from ba.metadata.derived import get_derived
from ba.metadata.populate import populate_speaking_rate
from ba.utils.audio_utils import audio_file
from ba.utils.display_utils import plural


def metadata(benchmark: BenchmarkOption = None) -> None:
    """Derive again the metadata that is worked out here, and save it with the test cases.

    `ba fetch` fills every field as it reads a benchmark, so this is for after a change to how
    a field is derived. The fields measured on the audio are measured where the audio is on
    disk (ba fetch --audio), and are left as they are where it is not.
    """
    with guard():
        paths, cfg, store = setup()

        # The benchmarks named. With none named: every benchmark that has its list.
        _, benchmark_ids = picked(cfg, None, benchmark)
        if not benchmark:
            benchmark_ids = [benchmark_id for benchmark_id in benchmark_ids if store.test_cases.has(benchmark_id)]

        for benchmark_id in benchmark_ids:
            test_cases = store.test_cases.read(benchmark_id)
            references = store.references.read(benchmark_id)  # by what is scored: a test case, or its session
            with_audio = 0
            for test_case in test_cases:
                audio = audio_file(paths, test_case)
                if audio is not None:
                    with_audio += 1
                for field, value in get_derived(references[test_case.unit_id], audio, paths).items():
                    setattr(test_case, field, value)
            populate_speaking_rate(test_cases)  # from the words just counted
            store.test_cases.write(benchmark_id, test_cases)
            typer.echo(f"{benchmark_id}: {plural(len(test_cases), 'test case')}, {with_audio} with audio on disk")
