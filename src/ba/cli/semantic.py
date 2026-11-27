"""``ba semantic``: compute semantic error, and store it beside WER and CER."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from ba.actions import ablation, scoring
from ba.cli.common import (
    BenchmarkOption,
    SystemOption,
    guard,
    picked,
    setup,
)
from ba.metrics import semantic as semantic_metric
from ba.normalize.registry import FAMILIES_OF, normalizers
from ba.utils.display_utils import plural


def semantic(
    system: SystemOption = None,
    benchmark: BenchmarkOption = None,
    normalizer: Annotated[
        list[str] | None,
        typer.Option("--normalizer", help="Repeat for several. Default: every normalizer."),
    ] = None,
    families: Annotated[
        bool,
        typer.Option("--families", help="Also with each rule family of own and leaderboard_split switched off in turn, for the family table."),
    ] = False,
    recompute: Annotated[
        bool,
        typer.Option("--recompute", help="Compute again even where it is stored."),
    ] = False,
) -> None:
    """Compute semantic error and store it beside WER and CER."""
    with guard():
        paths, cfg, store = setup()
        registry = normalizers(paths)
        normalizer_ids = normalizer or registry.ids()
        chosen_normalizers = [registry.get(normalizer_id) for normalizer_id in normalizer_ids]
        if families:
            made_of_families = [normalizer_id for normalizer_id in normalizer_ids if normalizer_id in FAMILIES_OF]
            if not made_of_families:
                raise ValueError(f"--families is for a normalizer made of rule families. Choose one of: {', '.join(FAMILIES_OF)}")
            for normalizer_id in made_of_families:
                # The first variant is the normalizer itself, which is chosen already.
                switched_off = list(ablation.variants(paths, normalizer_id).values())[1:]
                chosen_normalizers.extend(switched_off)
        system_ids, benchmark_ids = picked(cfg, system, benchmark)

        embedder = None  # the encoder is loaded only once there is something to compute
        done = 0
        for benchmark_id in benchmark_ids:
            runs = store.hypotheses.runs(system_ids, [benchmark_id])
            if runs and store.test_cases.scored_per_session(benchmark_id):
                typer.echo(f"{benchmark_id}: left out. It is scored per session, and a whole session is far more text than a sentence encoder reads.")
                continue
            # The benchmark's embeddings, shared by every normalizer scored on it: a text two of them leave alike is embedded once.
            embeddings: dict[str, Any] = {}
            for chosen in chosen_normalizers:
                to_do = []
                for run in runs:
                    if recompute or not store.scores.has(run, chosen.id, "semantic"):
                        to_do.append(run)
                if not to_do:
                    continue
                if embedder is None:
                    embedder = semantic_metric.SentenceTransformerEmbedder()
                scoring.score_semantic(store, to_do, chosen, embedder, embeddings)
                done += len(to_do)
                typer.echo(f"{benchmark_id}, {chosen.id} normalizer: {plural(len(to_do), 'model')}")

        if embedder is None:
            typer.echo("Nothing to compute: semantic error is stored for every model and normalizer chosen.")
        else:
            typer.echo(f"Semantic error computed for {done} model and normalizer pairs with {semantic_metric.MODEL}. See it with: ba score --metric semantic")
