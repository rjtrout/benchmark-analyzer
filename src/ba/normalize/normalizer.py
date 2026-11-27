"""A normalizer: a text-to-text function with an id.

One thing a normalizer may forgive cannot be done to a text on its own, since it takes both
texts to see it: a compound that is split in one and joined in the other. So a normalizer
says whether it forgives that, and the word aligner is told.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Normalizer:
    id: str
    fn: Callable[[str], str]
    # A compound split in one text and joined in the other is no error: "white paper" against
    # "whitepaper". The leaderboard's normalizer forgives it. Where its rules are families, it is the compounds family.
    forgives_compounds: bool = False

    def __call__(self, text: str) -> str:
        return self.fn(text)
