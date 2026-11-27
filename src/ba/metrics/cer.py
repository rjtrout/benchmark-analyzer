"""Character errors: the counts for one test case. A control beside WER, not a measure of meaning."""

from __future__ import annotations

import regex
from kaldialign import edit_distance

from ba.normalize.normalizer import Normalizer


def characters(text: str) -> tuple[str, ...]:
    """One token per character, with spaces left out, as the leaderboard counts characters."""
    return tuple(regex.findall(r"\X", "".join(text.split())))


def counts(ref: str, hyp: str, normalizer: Normalizer) -> dict[str, int]:
    """A test case's character error counts: reference characters, substitutions, deletions and insertions."""
    result = edit_distance(characters(ref), characters(hyp), merge_compounds=False)
    return {
        "n_ref": int(result["ref_len"]),
        "n_sub": int(result["sub"]),
        "n_del": int(result["del"]),
        "n_ins": int(result["ins"]),
    }
