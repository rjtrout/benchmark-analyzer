"""Our own normalizer: the leaderboard's rules and four more that are ours, each a family that switches off.

The leaderboard's rules come first, family by family, as ``leaderboard_split`` has them. Ours
follow. They forgive what a word-for-word transcript keeps and a tidied one leaves out:

cut-off-words      A word that was cut off is dropped: "th the same" is "the same".
discourse-markers  "you know" and "i mean" are dropped.
interjections      Oh, yeah, yep, yup, wow, hey, ooh, aw and whoa are dropped.
repetitions        A word said twice in a row counts once: "i i think" is "i think".

With its four families off, this gives exactly what ``leaderboard_split`` gives. Switching
one family off shows what that family alone forgives, which is what ``ba ablate`` reports.

Where each rule works. A cut-off word is looked for in the text as written, before the
leaderboard's rules: those set a possessive's s apart from its word, and it would be taken
for one. The other three work on the text as the leaderboard's rules leave it, lowercase and
without punctuation. Capitals and punctuation are given back at the end, where their
families are off.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ba.normalize import leaderboard_split
from ba.normalize.leaderboard_split import SplitSteps, transfer
from ba.normalize.normalizer import Normalizer
from ba.paths import Paths
from ba.utils.list_utils import unique

# The leaderboard's thirteen families, and then ours.
FAMILIES: dict[str, str] = {
    **leaderboard_split.FAMILIES,
    "cut-off-words": "A word that was cut off is dropped: th the same is the same.",
    "discourse-markers": "The markers you know and i mean are dropped.",
    "interjections": "Interjections such as oh, yeah and wow are dropped.",
    "repetitions": "A word said twice in a row counts once: i i think is i think.",
}

_LONE_CONSONANT = re.compile(r"[b-df-hj-np-tv-z]")
_CUT_OFF = re.compile(r"[b-df-hj-np-tv-z]|[cstw]h")  # a lone consonant, or ch, sh, th or wh
_EDGES = re.compile(r"^[^\w']+|[^\w']+$")  # what stands at a word's edges, but for an apostrophe: 's is part of a word
_MARKERS = re.compile(r"\b(you know|i mean)\b")
_INTERJECTIONS = re.compile(r"\b(oh|yeah|yep|yup|wow|hey|ooh|aw|whoa)\b")


def drop_cut_off_words(text: str) -> str:
    """The text as written without its cut-off words: a lone consonant, or ch, sh, th or wh, standing alone.

    One with another lone letter or digit beside it is left. That is something spelled out,
    such as "U S" or "5 G", and the initialisms rule joins it.
    """
    words = text.split()
    bare = [_EDGES.sub("", word).lower() for word in words]  # no capitals, and nothing at the edges
    kept = []
    for position, word in enumerate(words):
        one_before = position > 0 and len(bare[position - 1]) == 1
        one_after = position + 1 < len(words) and len(bare[position + 1]) == 1
        if _CUT_OFF.fullmatch(bare[position]) and not one_before and not one_after:
            continue
        kept.append(word)
    return " ".join(kept)


def drop_repetitions(text: str) -> str:
    """The text with a word said twice in a row said once.

    A number said twice is two numbers, and a letter spelled twice is two letters: "50 50"
    and "b b c" stay as they are.
    """
    kept: list[str] = []
    for word in text.split():
        said_again = bool(kept) and word == kept[-1]
        is_number = any(character.isdigit() for character in word)
        is_letter = _LONE_CONSONANT.fullmatch(word) is not None
        if said_again and not is_number and not is_letter:
            continue
        kept.append(word)
    return " ".join(kept)


class OwnSteps:
    """The text function of ``own``: the leaderboard's steps and then ours, each behind its family's switch."""

    def __init__(self, paths: Paths, on: frozenset[str]) -> None:
        self.on = on
        # The leaderboard's rules always give lowercase text without punctuation here. Our rules
        # need it so, and capitals and punctuation are given back after them, below.
        theirs = (on & frozenset(leaderboard_split.FAMILIES)) | {"case", "punctuation"}
        self._leaderboard_rules = SplitSteps(paths, theirs)

    def __call__(self, original: str) -> str:
        on = self.on
        if "cut-off-words" in on:
            original = drop_cut_off_words(original)
        text = self._leaderboard_rules(original)
        if "discourse-markers" in on:
            text = re.sub(r"\s+", " ", _MARKERS.sub(" ", text))
        if "interjections" in on:
            text = re.sub(r"\s+", " ", _INTERJECTIONS.sub(" ", text))
        if "repetitions" in on:
            text = drop_repetitions(text)
        keep_case = "case" not in on
        keep_punctuation = "punctuation" not in on
        if keep_case or keep_punctuation:
            text = transfer(original, text, case=keep_case, punctuation=keep_punctuation)
        return text


@lru_cache(maxsize=64)
def _steps(paths: Paths, on: frozenset[str]) -> OwnSteps:
    return OwnSteps(paths, on)


def own_normalizer(paths: Paths, without: tuple[str, ...] | list[str] = ()) -> Normalizer:
    """Our own normalizer, or a variant of it with some of the families switched off."""
    without = tuple(unique(without))
    unknown = [family for family in without if family not in FAMILIES]
    if unknown:
        raise ValueError(f"Unknown rule family: {', '.join(unknown)}. Families: {', '.join(FAMILIES)}")
    on = frozenset(FAMILIES) - set(without)
    if "numerals" not in on:
        on -= {"ordinals"}  # ordinals are handled inside the number rules
    normalizer_id = "own"
    if without:
        normalizer_id += "-without-" + "-".join(without)
    return Normalizer(
        id=normalizer_id,
        fn=_steps(paths, on),
        forgives_compounds="compounds" in on,
    )
