"""The pair suite: what each normalizer treats as the same words, and what it must keep apart.

Every pair is run against every normalizer. The table is the test, and it is also the short
answer to "what does each normalizer forgive". A pair marked for a normalizer must come out
equal under it, and every other normalizer must keep the pair apart.
"""

from __future__ import annotations

import pytest

from ba.metrics import wer
from ba.normalize.registry import normalizers

ALL = ("verbatim", "case-punct", "whisper-stock", "leaderboard", "leaderboard_split", "own")
FORK = ("leaderboard", "leaderboard_split", "own")  # the leaderboard's additions to Whisper's rules
RULE_BASED = ("whisper-stock", *FORK)  # every rule-based normalizer
OWN = ("own",)  # what our own adds to the leaderboard's rules

# (first, second, the normalizers under which the two must match)
MUST_MATCH = [
    ("Hello, World!", "hello world", ("case-punct", *RULE_BASED)),
    ("well-known", "well known", ("case-punct", *RULE_BASED)),
    ("twenty two", "22", RULE_BASED),
    ("the first", "the 1st", RULE_BASED),
    ("1,234", "one thousand two hundred thirty four", RULE_BASED),
    ("three point five", "3.5", RULE_BASED),
    ("$20", "twenty dollars", RULE_BASED),
    ("50%", "fifty percent", RULE_BASED),
    ("colour", "color", RULE_BASED),
    ("I can't", "I can not", RULE_BASED),
    ("it's here", "it is here", RULE_BASED),
    ("gonna go", "going to go", RULE_BASED),
    ("uh I see", "I see", RULE_BASED),
    ("mm-hmm yes", "yes", RULE_BASED),
    ("Dr. Smith", "doctor smith", RULE_BASED),
    ("[laughter] yes", "yes", RULE_BASED),
    # What the leaderboard's fork adds to the stock Whisper normalizer.
    ("okay", "ok", FORK),
    ("I cannot", "I can not", FORK),
    ("the b b c", "the bbc", FORK),
    ("on wi-fi", "on wifi", FORK),
    ("white paper", "whitepaper", FORK),
    ("at 2:00 PM", "at two pm", FORK),
    ("alan", "allen", FORK),
    # What our own adds to the leaderboard's rules.
    ("th the cat sat", "the cat sat", OWN),
    ("you know we agree", "we agree", OWN),
    ("wow we agree", "we agree", OWN),
    ("we we agree", "we agree", OWN),
]

# Pairs with different words. No normalizer should call them the same.
MUST_DIFFER = [
    ("I do", "I don't"),
    ("fifteen", "fifty"),
    ("the cat", "the cats"),
    ("no", "know"),
]

# Collisions: different words that a normalizer nonetheless treats as the same. Each is a
# known cost of rewriting both sides to one form, kept here so that a change shows up.
COLLISIONS = [
    ("oh", "zero", ("whisper-stock",)),  # stock Whisper reads every "oh" as a digit
    ("U.S.", "us", FORK),  # the country and the pronoun
]


def same(normalizer, first: str, second: str) -> bool:
    counts = wer.counts(normalizer(first), normalizer(second), normalizer)
    return counts["n_sub"] + counts["n_del"] + counts["n_ins"] == 0


CASES = [(a, b, matching) for a, b, matching in MUST_MATCH + COLLISIONS] + [(a, b, ()) for a, b in MUST_DIFFER]


@pytest.mark.parametrize("normalizer_id", ALL)
@pytest.mark.parametrize(("first", "second", "matching"), CASES, ids=[f"{a} ~ {b}" for a, b, _ in CASES])
def test_pair(paths, normalizer_id, first, second, matching):
    assert same(normalizers(paths).get(normalizer_id), first, second) == (normalizer_id in matching)


def test_verbatim_forgives_nothing(paths):
    verbatim = normalizers(paths).get("verbatim")
    assert not any(same(verbatim, a, b) for a, b, _ in MUST_MATCH)
    assert same(verbatim, "exactly this", "exactly this")
