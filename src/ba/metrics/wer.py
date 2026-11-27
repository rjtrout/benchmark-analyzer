"""Word errors: the counts for one test case, from the aligner the leaderboard uses."""

from __future__ import annotations

from kaldialign import edit_distance

from ba.normalize.normalizer import Normalizer


def counts(ref: str, hyp: str, normalizer: Normalizer) -> dict[str, int]:
    """Reference words, substitutions, deletions and insertions for one test case.

    Both texts are as the normalizer leaves them, and are split on whitespace. Where the
    normalizer forgives compounds, one that is split in one text and joined in the other is
    no error. Counts are stored, not a rate, so that the overall figure, and the mean and
    the median over test cases, are queries over them.
    """
    reference_words = tuple(ref.split())
    transcript_words = tuple(hyp.split())
    result = edit_distance(reference_words, transcript_words, merge_compounds=normalizer.forgives_compounds)
    return {
        "n_ref": int(result["ref_len"]),
        "n_sub": int(result["sub"]),
        "n_del": int(result["del"]),
        "n_ins": int(result["ins"]),
    }

