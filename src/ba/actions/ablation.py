"""What each rule family of a normalizer is worth to a run.

Two normalizers are made of rule families: ``leaderboard_split``, the leaderboard's rules,
and ``own``, those rules and four more. A family switched off is a normalizer like any
other: ``own-without-numerals`` is ``own`` with its number rules off. So the family table is
the score table of those normalizers, with each figure's change from the run's figure under
the whole normalizer. The first time is slow on a large set, a scoring for each family of
each model and benchmark, and each is saved.

Word and character errors are counted as the table is asked for. Semantic error is read: it
is computed for the families by ``ba semantic --families``.
"""

from __future__ import annotations

from typing import Any

from ba.actions.score_summary import score_table
from ba.aggregations.registry import AGGREGATIONS
from ba.normalize.normalizer import Normalizer
from ba.normalize.registry import FAMILIES_OF, with_families
from ba.paths import Paths
from ba.store.store import Store
from ba.store.transcriptions import Run

EVERY_FAMILY_ON = "all families on"  # the variant every other one is compared with: the whole normalizer


def variants(paths: Paths, normalizer_id: str) -> dict[str, Normalizer]:
    """A normalizer's variants, each under what the table calls it.

    Every family on, then each family switched off in turn. ``normalizer_id`` is ``own`` or
    ``leaderboard_split``.
    """
    found = {EVERY_FAMILY_ON: with_families(paths, normalizer_id)}
    for family in FAMILIES_OF[normalizer_id]:
        found[f"without {family}"] = with_families(paths, normalizer_id, without=[family])
    return found


def ablate(
    store: Store,
    runs: list[Run],
    normalizer_id: str,
    metric_id: str = "wer",
    agg: str = "overall",
    recompute: bool = False,
) -> list[dict[str, Any]]:
    """The family table of a normalizer: a row per run and variant, with the run's figure and its change.

    The change is in points against the run's figure with every family on. A positive change
    means the family was forgiving that much error: it is how much the family helps the
    score. ``recompute`` counts every variant again, from the transcripts as written.

    A run is in the table when it has the metric under every variant. Semantic error is
    only read here, so a run that does not have it for the families yet is left out.
    """
    AGGREGATIONS.get(agg)  # an unknown aggregation is an error before any work is done
    named = variants(store.paths, normalizer_id)
    count_again = recompute and metric_id != "semantic"
    scored = score_table(store, runs, list(named.values()), [agg], rescore=count_again, metric_ids=[metric_id])

    rows_of_run: dict[str, int] = {}  # run id -> how many of the variants it has a row for
    for row in scored:
        rows_of_run[row["run_id"]] = rows_of_run.get(row["run_id"], 0) + 1
    table = [row for row in scored if rows_of_run[row["run_id"]] == len(named)]

    variant_of = {}  # a normalizer's id -> what the table calls it
    for name, normalizer in named.items():
        variant_of[normalizer.id] = name
    with_every_family_on = {}  # run id -> the run's figure under the whole normalizer
    for row in table:
        if variant_of[row["normalizer"]] == EVERY_FAMILY_ON:
            with_every_family_on[row["run_id"]] = row[agg]
    for row in table:
        row["variant"] = variant_of[row["normalizer"]]
        row["change"] = row[agg] - with_every_family_on[row["run_id"]]
    return table
