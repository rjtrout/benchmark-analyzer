"""The normalizers, from the text as written to the most forgiving.

verbatim           no normalization
case-punct         the floor: case and punctuation only
whisper-stock      OpenAI Whisper's English normalizer, as shipped
leaderboard        the Open ASR Leaderboard's own, unchanged
leaderboard_split  the leaderboard's rules as families that switch off (see ``leaderboard_split.py``)
own                those families and four more of ours (see ``own.py``)

The last two are made of rule families: ``with_families`` gives either, or a variant of it
with some families switched off. To add a normalizer, write its function in a module here
and register it below.
"""

from __future__ import annotations

from functools import lru_cache

from ba.normalize import basic, leaderboard, leaderboard_split, own, whisper_stock
from ba.normalize.leaderboard_split import leaderboard_split_normalizer
from ba.normalize.normalizer import Normalizer
from ba.normalize.own import own_normalizer
from ba.paths import Paths
from ba.utils.registry import Registry

# The normalizers that are made of rule families, and the families of each.
FAMILIES_OF = {"own": own.FAMILIES, "leaderboard_split": leaderboard_split.FAMILIES}


def with_families(paths: Paths, normalizer_id: str, without: tuple[str, ...] | list[str] = ()) -> Normalizer:
    """A normalizer that is made of rule families, or a variant of it with some of them switched off."""
    if normalizer_id == "own":
        return own_normalizer(paths, without)
    if normalizer_id == "leaderboard_split":
        return leaderboard_split_normalizer(paths, without)
    raise ValueError(f"The {normalizer_id} normalizer is not made of rule families. Choose one of: {', '.join(FAMILIES_OF)}")


@lru_cache(maxsize=8)
def normalizers(paths: Paths) -> Registry:
    """Every normalizer, by id. ``paths`` says where the leaderboard's code is."""
    registry = Registry("normalizers")
    registry.register("verbatim", Normalizer(id="verbatim", fn=basic.verbatim))
    registry.register("case-punct", Normalizer(id="case-punct", fn=basic.case_punct))
    registry.register("whisper-stock", Normalizer(id="whisper-stock", fn=whisper_stock.EnglishTextNormalizer()))
    registry.register(
        "leaderboard",
        Normalizer(id="leaderboard", fn=leaderboard.english_normalizer(paths), forgives_compounds=True),
    )
    registry.register("leaderboard_split", leaderboard_split_normalizer(paths))
    registry.register("own", own_normalizer(paths))
    return registry
