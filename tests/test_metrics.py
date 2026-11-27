"""The metrics: word and character counts, checked against the leaderboard's own scorer."""

from __future__ import annotations

from ba.metrics import wer
from ba.metrics.registry import METRICS
from ba.normalize import leaderboard
from ba.normalize.basic import verbatim as as_written
from ba.normalize.leaderboard_split import leaderboard_split_normalizer
from ba.normalize.normalizer import Normalizer
from ba.normalize.own import own_normalizer
from ba.normalize.registry import normalizers
from tests.conftest import leaderboards_score

# Two normalizers that leave a text as it is. One forgives a compound that is split in one text and joined in the other.
PLAIN = Normalizer(id="plain", fn=as_written)
FORGIVING = Normalizer(id="forgiving", fn=as_written, forgives_compounds=True)

TRICKY = [
    ("Mr. Smith won't pay $20, uh, okay?", "mister smith will not pay twenty dollars ok"),
    ("The white paper is on wi-fi.", "the whitepaper is on wifi"),
    ("I said [laughter] nothing at all", ""),
    ("She has twenty two apples", "she has 22 apples and pears"),
    ("colour and flavour", "color and flavor"),
    ("it's the B B C at 2:00 PM", "it is the bbc at two pm"),
    ("okay", "oh kay then"),
]


# ── the leaderboard's scorer, by construction ────────────────────────────────────────────


def test_leaderboard_normalizer_is_the_clones_normalizer(paths):
    leaderboard.ensure_importable(paths)
    from normalizer.normalizer import EnglishTextNormalizer

    theirs = EnglishTextNormalizer()
    ours = normalizers(paths).get("leaderboard")
    for ref, hyp in TRICKY:
        assert ours(ref) == theirs(ref)
        assert ours(hyp) == theirs(hyp)
    assert ours.forgives_compounds  # as the leaderboard's scorer does


def test_per_test_case_counts_sum_to_the_clones_own_manifest_score(paths):
    manifest = [
        {"audio_filepath": f"c{i}", "duration": 2.0, "time": 0.1, "text": ref, "pred_text": hyp}
        for i, (ref, hyp) in enumerate(TRICKY)
    ]
    theirs = leaderboards_score(paths, manifest)
    ours = normalizers(paths).get("leaderboard")
    counts = [wer.counts(ours(m["text"]), ours(m["pred_text"]), ours) for m in manifest]
    assert sum(c["n_ins"] for c in counts) == theirs["ins"]
    assert sum(c["n_del"] for c in counts) == theirs["del"]
    assert sum(c["n_sub"] for c in counts) == theirs["sub"]
    errors = sum(c["n_ins"] + c["n_del"] + c["n_sub"] for c in counts)
    assert round(100 * errors / sum(c["n_ref"] for c in counts), 2) == theirs["wer"]
    assert errors > 0  # the fixture has real errors, so the equality means something


def test_verbatim_counts_case_punctuation_and_number_format(paths):
    verbatim = normalizers(paths).get("verbatim")
    assert verbatim("Hello, World!") == "Hello, World!"
    counts = wer.counts(verbatim("Hello, World! 22"), verbatim("hello world twenty two"), verbatim)
    assert counts == {"n_ref": 3, "n_sub": 3, "n_del": 0, "n_ins": 1}


def test_an_empty_hypothesis_is_all_deletions():
    assert wer.counts("three small words", "", PLAIN) == {"n_ref": 3, "n_sub": 0, "n_del": 3, "n_ins": 0}
    assert wer.counts("", "two words", PLAIN) == {"n_ref": 0, "n_sub": 0, "n_del": 0, "n_ins": 2}


def errors_of(counts: dict[str, int]) -> int:
    return counts["n_sub"] + counts["n_del"] + counts["n_ins"]


def test_a_split_compound_is_no_error_where_the_normalizer_forgives_compounds(paths):
    split, joined = "the white paper is here", "the whitepaper is here"
    # A normalizer that forgives compounds has the aligner match the two. One that does not counts a word written
    # for another and a word left out.
    assert errors_of(wer.counts(split, joined, FORGIVING)) == 0 and errors_of(wer.counts(split, joined, PLAIN)) == 2
    # The leaderboard's forgives them, as its scorer does. So does ours, where it is the compounds family, which
    # switches off. The three plainer ones never do.
    registry = normalizers(paths)
    assert not own_normalizer(paths, without=["compounds"]).forgives_compounds
    assert not leaderboard_split_normalizer(paths, without=["compounds"]).forgives_compounds
    for normalizer_id, errors in (("leaderboard", 0), ("leaderboard_split", 0), ("own", 0), ("whisper-stock", 2), ("case-punct", 2), ("verbatim", 2)):
        normalizer = registry.get(normalizer_id)
        assert errors_of(wer.counts(normalizer(split), normalizer(joined), normalizer)) == errors, normalizer_id


def test_cer_counts_characters_and_ignores_spaces(paths):
    verbatim = normalizers(paths).get("verbatim")
    counts = METRICS.get("cer")("abc def", "abd  def", verbatim)
    assert counts == {"n_ref": 6, "n_sub": 1, "n_del": 0, "n_ins": 0}
