"""The leaderboard's normalizer, taken apart into families that switch off.

With every family on, this gives exactly the leaderboard's normalizer, step for step. That
is checked on every reference and transcript that is stored. Switching one
family off shows what that family alone forgives, which is what ``ba ablate`` reports.
Our own normalizer, in ``own.py``, is these families and four more.

The steps and their tables (fillers, contractions, number words, spellings, names,
compounds) are the clone's own. Three families have no step to skip, so they work
differently:

case, punctuation  The rules need lowercase, unpunctuated text to work at all. So the
                   pipeline runs as usual, and afterwards each word the rules left alone
                   gets its capitals or its attached punctuation back (``transfer``).
hyphens            A hyphen inside a word is shielded from the pipeline and put back, so
                   ``well-known`` stays one word.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from kaldialign import align

from ba.normalize import leaderboard
from ba.normalize.normalizer import Normalizer
from ba.paths import Paths
from ba.utils.list_utils import unique

FAMILIES: dict[str, str] = {
    "case": "Capital letters do not matter.",
    "punctuation": "Punctuation marks do not matter.",
    "hyphens": "A hyphen separates words: well-known is well known.",
    "tags": "Non-speech tags such as [laughter] are dropped.",
    "fillers": "Fillers such as uh and um are dropped.",
    "contractions": "Contractions are expanded: can't is can not.",
    "titles": "Abbreviated titles are spelled out: dr is doctor.",
    "numerals": "Spelled-out numbers become digits, with money, percent and decimals.",
    "ordinals": "Ordinal words become digits with a suffix: first is 1st. Part of numerals.",
    "compounds": "Split and joined compounds match: wi fi is wifi, white paper is whitepaper.",
    "regional-spellings": "British spellings become American: colour is color.",
    "alternate-spellings": "Name variants share one spelling: alan is allen.",
    "initialisms": "Spelled-out initialisms are joined: b b c is bbc.",
}

_SHIELD = "zqhyphenqz"  # letters only, so that it passes through every step untouched
_INNER_HYPHEN = re.compile(r"(?<=\w)[-‐‑](?=\w)")
_EDGES = re.compile(r"^\W+|\W+$")
_EPS = "\x00"


def transfer(original: str, normalized: str, *, case: bool, punctuation: bool) -> str:
    """Give words the rules left alone their capitals and/or attached punctuation back.

    The original's words are aligned with the normalized ones. Where a word came through
    unchanged apart from case and edge punctuation, the original form is restored. A word
    the rules rewrote stays as they wrote it.
    """
    original_words = original.split()
    bare_words = [_EDGES.sub("", word).lower() for word in original_words]  # no capitals, no edge punctuation
    normalized_words = normalized.split()

    restored = []
    position = 0  # of the next original word
    for bare, target in align(bare_words, normalized_words, _EPS):
        original_word = None
        if bare != _EPS:
            original_word = original_words[position]
            position += 1
        if target == _EPS:
            continue  # the rules dropped this word
        rewritten = original_word is None or bare != target or bare == ""
        if rewritten:
            restored.append(target)
        elif case and punctuation:
            restored.append(original_word)
        elif case:
            restored.append(_EDGES.sub("", original_word))
        else:
            # Punctuation only: the normalized word, between the marks the original had at its edges.
            core = _EDGES.sub("", original_word)
            start = original_word.find(core)
            before, after = original_word[:start], original_word[start + len(core) :]
            restored.append(before + target + after)
    return " ".join(restored)


class SplitSteps:
    """The text function of ``leaderboard_split``: the leaderboard's steps, each behind its family's switch."""

    def __init__(self, paths: Paths, on: frozenset[str]) -> None:
        leaderboard.ensure_importable(paths)
        from normalizer import normalizer as theirs

        self.on = on
        self._theirs = theirs
        base = theirs.EnglishTextNormalizer()
        self._fillers = base.ignore_patterns
        self._replacers = []  # each of the leaderboard's replacement rules, with the family it belongs to
        for pattern, replacement in base.replacers.items():
            self._replacers.append((pattern, replacement, replacer_family(pattern, replacement)))
        self._compounds = base.compound_words
        self._spellings = base.standardize_spellings
        self._names = base.standardize_names
        self._initialisms = base.standardize_acronyms
        self._numbers = base.standardize_numbers
        if "ordinals" not in on:
            self._numbers = _without_ordinals(theirs)

    def __call__(self, original: str) -> str:
        """The leaderboard's steps, in its order. A step runs only if its family is switched on."""
        on, theirs = self.on, self._theirs
        text = original.lower()
        if "hyphens" not in on:
            text = _INNER_HYPHEN.sub(_SHIELD, text)
        if "tags" in on:
            text = theirs.remove_non_speech_annotations(text)
        if "fillers" in on:
            text = re.sub(self._fillers, "", text)
        text = re.sub(r"\s+'", "'", text)
        for pattern, replacement, family in self._replacers:
            if family in on:
                text = re.sub(pattern, replacement, text)
        if "numerals" in on:
            text = re.sub(r"(\d),(\d)", r"\1\2", text)
        text = re.sub(r"\.([^0-9]|$)", r" \1", text)
        text = theirs.remove_symbols_and_diacritics(text, keep=".%$¢€£")
        if "compounds" in on:
            for pattern, replacement in self._compounds.items():
                text = re.sub(pattern, replacement, text)
        if "numerals" in on:
            text = self._numbers(text)
        if "regional-spellings" in on:
            text = self._spellings(text)
        if "alternate-spellings" in on:
            text = self._names(text)
        if "initialisms" in on:
            text = self._initialisms(text)
        text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
        text = re.sub(r"([^0-9])%", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)
        if "hyphens" not in on:
            text = text.replace(_SHIELD, "-")
        keep_case = "case" not in on
        keep_punctuation = "punctuation" not in on
        if keep_case or keep_punctuation:
            text = transfer(original, text, case=keep_case, punctuation=keep_punctuation)
        return text


def replacer_family(pattern: str, replacement: str) -> str:
    """Which family one of the clone's replacement rules belongs to."""
    if pattern.startswith(r"\b(\d"):
        return "numerals"  # bare o'clock times: "2:00" is "2"
    if re.fullmatch(r"\\b[a-z]+\\b", pattern) and replacement.endswith(" "):
        return "titles"  # "\bdr\b" -> "doctor "
    return "contractions"


def _without_ordinals(theirs: Any):
    """The clone's number normalizer with the ordinal words taken out of its vocabulary."""

    class NoOrdinals(theirs.EnglishNumberNormalizer):
        def __init__(self) -> None:
            super().__init__()
            ordinals = set(self.ones_ordinal) | set(self.tens_ordinal) | set(self.multipliers_ordinal)
            self.ones_suffixed = dict(self.ones_plural)
            self.tens_suffixed = dict(self.tens_plural)
            self.multipliers_suffixed = dict(self.multipliers_plural)
            self.words = self.words - ordinals

    return NoOrdinals()


@lru_cache(maxsize=64)
def _steps(paths: Paths, on: frozenset[str]) -> SplitSteps:
    return SplitSteps(paths, on)


def leaderboard_split_normalizer(paths: Paths, without: tuple[str, ...] | list[str] = ()) -> Normalizer:
    """The leaderboard's rules as families, or a variant with some of the families switched off."""
    without = tuple(unique(without))
    unknown = [family for family in without if family not in FAMILIES]
    if unknown:
        raise ValueError(f"Unknown rule family: {', '.join(unknown)}. Families: {', '.join(FAMILIES)}")
    on = frozenset(FAMILIES) - set(without)
    if "numerals" not in on:
        on -= {"ordinals"}  # ordinals are handled inside the number rules
    normalizer_id = "leaderboard_split"
    if without:
        normalizer_id += "-without-" + "-".join(without)
    return Normalizer(
        id=normalizer_id,
        fn=_steps(paths, on),
        forgives_compounds="compounds" in on,
    )
