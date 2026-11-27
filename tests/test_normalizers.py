"""The normalizers: the stock Whisper one, the floor, the leaderboard's rules as families, and our own with four more."""

from __future__ import annotations

import pytest

from ba.metrics import wer
from ba.normalize import leaderboard_split, own
from ba.normalize.leaderboard_split import leaderboard_split_normalizer, transfer
from ba.normalize.own import own_normalizer
from ba.normalize.registry import normalizers

TEXTS = [
    "Mr. Smith won't pay $20, uh, okay?",
    "The white paper is on wi-fi.",
    "I said [laughter] nothing at all",
    "She has twenty two apples and 3.5% of the 1st prize",
    "colour and flavour, etc. at 2:00 PM",
    "it's the B B C, y'all -- Dr. O'Neil's well-known café",
    "Um, I'd've gone; she's been there. Alan and Allan!",
    "one oh one, double seven, and a half... 1,234 dollars",
    "So, you know, we we are, oh, encouraged. I mean, yeah.",
    "We m uh we must stay, th the same",
    "",
    "   ",
]


def matches(normalizer, first: str, second: str) -> bool:
    counts = wer.counts(normalizer(first), normalizer(second), normalizer)
    return counts["n_sub"] + counts["n_del"] + counts["n_ins"] == 0


def test_leaderboard_split_with_every_family_on_is_the_leaderboards_normalizer(paths):
    leaderboard, split = normalizers(paths).get("leaderboard"), normalizers(paths).get("leaderboard_split")
    for text in TEXTS:
        assert split(text) == leaderboard(text), text
    assert split.forgives_compounds and leaderboard.forgives_compounds


def test_own_with_its_four_families_off_is_leaderboard_split(paths):
    split = normalizers(paths).get("leaderboard_split")
    without_ours = own_normalizer(paths, without=["cut-off-words", "discourse-markers", "interjections", "repetitions"])
    for text in TEXTS:
        assert without_ours(text) == split(text), text
    # With them on it goes further than the leaderboard does.
    ours = normalizers(paths).get("own")
    assert split("So, you know, we we are, oh, encouraged.") == "so you know we we are oh encouraged"
    assert ours("So, you know, we we are, oh, encouraged.") == "so we are encouraged"
    assert ours.forgives_compounds


# One pair per family: equal with the family on, different with it off. The tests below take the
# families from the code, so a family with no pair here fails them by name.
PROBES = {
    "case": ("Hello World", "hello world"),
    "punctuation": ("hello, world.", "hello world"),
    "hyphens": ("a well-known fact", "a well known fact"),
    "tags": ("yes [laughter] indeed", "yes indeed"),
    "fillers": ("uh I think so", "I think so"),
    "contractions": ("I can't go", "I can not go"),
    "titles": ("dr. smith", "doctor smith"),
    "numerals": ("twenty two apples", "22 apples"),
    "ordinals": ("the first time", "the 1st time"),
    "compounds": ("the white paper", "the whitepaper"),
    "regional-spellings": ("the colour", "the color"),
    "alternate-spellings": ("alan said", "allen said"),
    "initialisms": ("the b b c", "the bbc"),
}
# The same for the four families that are ours.
OWN_PROBES = {
    "cut-off-words": ("we m must stay below", "we must stay below"),
    "discourse-markers": ("so you know we are encouraged", "so we are encouraged"),
    "interjections": ("oh that is good", "that is good"),
    "repetitions": ("i i think so", "i think so"),
}
EVERY_PROBE = {**PROBES, **OWN_PROBES}


def test_the_compound_merging_aligner_already_forgives_spelled_out_initialisms(paths):
    # A finding, not an accident: the leaderboard's scorer treats "b b c" against "bbc" as a
    # split compound, so its initialism rule changes nothing unless compounds are off too.
    first, second = PROBES["initialisms"]
    assert matches(leaderboard_split_normalizer(paths, without=["initialisms"]), first, second)
    assert not matches(leaderboard_split_normalizer(paths, without=["initialisms", "compounds"]), first, second)
    # The same holds in own: a word said twice counts once, and a letter spelled twice stays two letters.
    assert matches(own_normalizer(paths, without=["initialisms"]), first, second)
    assert not matches(own_normalizer(paths, without=["initialisms", "compounds"]), first, second)


@pytest.mark.parametrize("family", [f for f in leaderboard_split.FAMILIES if f != "initialisms"])
def test_switching_a_family_of_leaderboard_split_off_stops_it_forgiving_its_difference(paths, family):
    first, second = PROBES[family]
    assert matches(normalizers(paths).get("leaderboard_split"), first, second)
    without = leaderboard_split_normalizer(paths, without=[family])
    assert not matches(without, first, second)
    assert without.id == f"leaderboard_split-without-{family}"


@pytest.mark.parametrize("family", [f for f in own.FAMILIES if f != "initialisms"])
def test_switching_a_family_of_own_off_stops_it_forgiving_its_difference(paths, family):
    first, second = EVERY_PROBE[family]
    assert matches(normalizers(paths).get("own"), first, second)
    without = own_normalizer(paths, without=[family])
    assert not matches(without, first, second)
    assert without.id == f"own-without-{family}"


@pytest.mark.parametrize("family", list(leaderboard_split.FAMILIES))
def test_switching_one_family_of_leaderboard_split_off_leaves_the_others_working(paths, family):
    without = leaderboard_split_normalizer(paths, without=[family])
    for other, (first, second) in PROBES.items():
        # ordinals are part of numerals: with numerals off, ordinals are off too.
        if other == family or (family == "numerals" and other == "ordinals"):
            continue
        assert matches(without, first, second), f"{other} broke when {family} was switched off"


@pytest.mark.parametrize("family", list(own.FAMILIES))
def test_switching_one_family_of_own_off_leaves_the_others_working(paths, family):
    without = own_normalizer(paths, without=[family])
    for other, (first, second) in EVERY_PROBE.items():
        if other == family or (family == "numerals" and other == "ordinals"):
            continue
        assert matches(without, first, second), f"{other} broke when {family} was switched off"


def test_an_unknown_family_names_the_real_ones(paths):
    with pytest.raises(ValueError, match="numerals"):
        leaderboard_split_normalizer(paths, without=["numbers"])
    with pytest.raises(ValueError, match="repetitions"):
        own_normalizer(paths, without=["numbers"])
    # The four that are ours are no families of the leaderboard's rules.
    with pytest.raises(ValueError, match="Unknown rule family: repetitions"):
        leaderboard_split_normalizer(paths, without=["repetitions"])


def test_a_word_that_was_cut_off_is_dropped(paths):
    ours = normalizers(paths).get("own")
    # A lone consonant, or th, wh, sh or ch, with no other lone letter beside it.
    assert ours("We m must stay, th the same") == "we must stay the same"
    split = normalizers(paths).get("leaderboard_split")
    assert ours("one l and one dude off the street") == split("one and one dude off the street")
    # Letters that are spelled out stand beside one another, and are left for the initialisms rule.
    assert ours("the U S economy") == "the us economy"
    assert ours("it is a 5 G phone") == split("it is a 5 G phone")
    # A vowel alone is a word. And a possessive's s is part of its word as written, so it is no cut-off word.
    assert ours("I saw a dog") == "i saw a dog"
    assert ours("the company's results") == "the company s results"
    # A cut-off word, a filler and a restart, each gone by its own rule.
    assert ours("we m uh we must") == "we must"


def test_the_markers_and_the_interjections_are_dropped(paths):
    ours = normalizers(paths).get("own")
    assert ours("When we look at, you know, the base") == "when we look at the base"
    assert ours("I mean, it is, you know, fine") == "it is fine"
    assert ours("Oh, yeah, that is good, wow") == "that is good"
    # Only as whole words: "know" and "mean" alone are words like any other.
    assert ours("I know what you mean") == "i know what you mean"
    assert ours("oh ohio yeahs") == "ohio yeahs"


def test_a_word_said_twice_in_a_row_counts_once(paths):
    ours = normalizers(paths).get("own")
    assert ours("that that is very very very good") == "that is very good"
    assert ours("I I think so") == "i think so"
    # A number said twice is two numbers, and a letter spelled twice is two letters.
    assert ours("it was 50 50") == "it was 50 50"
    assert own_normalizer(paths, without=["initialisms"])("the b b c") == "the b b c"
    # A word that comes back after another word is not a repetition.
    assert ours("no it is no good") == "no it is no good"


def test_owns_rules_work_with_capitals_and_punctuation_kept(paths):
    kept = own_normalizer(paths, without=["case", "punctuation"])
    # The words the rules left alone get their capitals and their punctuation back.
    assert kept("So, you know, we are very very Encouraged.") == "So, we are very Encouraged."
    assert kept("We m must stay, th the same.") == "We must stay, the same."


def test_transfer_gives_back_case_and_punctuation_on_words_the_rules_left_alone():
    assert transfer("Hello, World! It's 22.", "hello world it is 22", case=True, punctuation=True) == "Hello, World! it is 22."
    assert transfer("Hello, World!", "hello world", case=True, punctuation=False) == "Hello World"
    assert transfer("Hello, World!", "hello world", case=False, punctuation=True) == "hello, world!"
    assert transfer("", "", case=True, punctuation=True) == ""


def test_case_punct_keeps_an_apostrophe_inside_a_word(paths):
    floor = normalizers(paths).get("case-punct")
    # Inside a word an apostrophe is spelling. Without it other words appear: we'll is not well.
    for one, other in [("we'll", "well"), ("we're", "were"), ("it's", "its"), ("she'll", "shell"), ("I'd", "id"), ("can't", "cant")]:
        assert not matches(floor, one, other), (one, other)
    assert floor("We'll go, won't we?") == "we'll go won't we"
    # The curly apostrophe is the same letter.
    assert matches(floor, "we’ll", "we'll")
    # At the edge of a word it is a quotation mark, or a plural's possessive, and goes as punctuation does.
    assert floor("'Hello,' she said. The dogs' bowls, rock 'n' roll.") == "hello she said the dogs bowls rock n roll"
    assert floor("O'Neil's 'tis ' '") == "o'neil's tis"


