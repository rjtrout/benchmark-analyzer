"""What each rule family is worth: a family switched off is a normalizer, and the family table is its scores beside the whole normalizer's."""

from __future__ import annotations

import numpy as np
import pytest

from ba.actions import ablation, scoring
from ba.actions.ablation import ablate
from ba.normalize import leaderboard_split, own
from ba.store.transcriptions import Run
from tests.conftest import TOY_RUN, make_run


def of_run(table, run=TOY_RUN):
    """One run's rows of a family table, by variant."""
    return {row["variant"]: row for row in table if row["run_id"] == run.run_id}


def test_the_variants_are_the_normalizer_and_the_normalizer_with_each_family_off_in_turn(paths):
    variants = ablation.variants(paths, "leaderboard_split")
    assert list(variants) == ["all families on", *[f"without {family}" for family in leaderboard_split.FAMILIES]]
    assert (variants["all families on"].id, variants["without fillers"].id) == ("leaderboard_split", "leaderboard_split-without-fillers")
    # Our own has the same thirteen families and four more.
    variants = ablation.variants(paths, "own")
    assert list(variants) == ["all families on", *[f"without {family}" for family in own.FAMILIES]]
    assert (variants["all families on"].id, variants["without repetitions"].id) == ("own", "own-without-repetitions")


def test_only_a_normalizer_made_of_families_has_a_family_table(paths):
    with pytest.raises(ValueError, match="own, leaderboard_split"):
        ablation.variants(paths, "leaderboard")


def written(paths) -> dict[str, int]:
    """Every file of scores, with when it was last written."""
    return {str(file): file.stat().st_mtime_ns for file in (paths.artifacts / "scores" / "ours").glob("*/*.parquet")}


def test_the_family_table_is_each_variants_score_and_its_change_from_the_whole_normalizers(paths):
    store = make_run(paths, statuses=["ok", "empty", "skipped"])
    table = ablate(store, [TOY_RUN], "leaderboard_split", "wer", "overall")
    assert [row["variant"] for row in table] == list(ablation.variants(paths, "leaderboard_split"))  # a row per variant, in their order
    first = table[0]
    assert (first["variant"], first["normalizer"], first["change"], first["ref_units"]) == ("all families on", "leaderboard_split", 0, 5)
    assert first["overall"] == pytest.approx(100 * 3 / 5)
    # A family switched off is a normalizer like any other: its scores are saved as any score is.
    assert store.scores.has(TOY_RUN, "leaderboard_split-without-fillers", "wer") and store.scores.has(TOY_RUN, "leaderboard_split", "wer")
    # Asked again, nothing is counted: the saved scores are read, for any aggregation.
    saved = written(paths)
    assert len(saved) == 14 and ablate(store, [TOY_RUN], "leaderboard_split", "wer", "overall") == table
    mean = ablate(store, [TOY_RUN], "leaderboard_split", "wer", "mean")
    assert mean[0]["mean"] == pytest.approx(100 * (1 / 3 + 1) / 2) and mean[0]["change"] == 0
    assert written(paths) == saved


def test_owns_family_table_has_its_seventeen_families(paths):
    store = make_run(paths)
    # The second transcript says a word twice and has a marker, where the reference has neither.
    store.hypotheses.write(TOY_RUN, {"toy/c0": "alpha beta gamma", "toy/c1": "delta you know epsilon epsilon"})
    rows = of_run(ablate(store, [TOY_RUN], "own", "wer", "overall"))
    assert list(rows) == list(ablation.variants(paths, "own")) and len(rows) == 18
    assert (rows["all families on"]["normalizer"], rows["all families on"]["overall"]) == ("own", 0.0)
    # With repetitions no longer forgiven the second "epsilon" is one word too many in five. The marker is two.
    assert rows["without repetitions"]["change"] == pytest.approx(100 * 1 / 5)
    assert rows["without discourse-markers"]["change"] == pytest.approx(100 * 2 / 5)
    assert rows["without fillers"]["change"] == 0
    assert rows["without repetitions"]["normalizer"] == "own-without-repetitions"
    # The leaderboard's rules forgive neither: under them the same transcript has three words too many.
    split = of_run(ablate(store, [TOY_RUN], "leaderboard_split", "wer", "overall"))
    assert split["all families on"]["overall"] == pytest.approx(100 * 3 / 5)


def test_a_family_switched_off_changes_the_score_by_what_it_forgave(paths):
    store = make_run(paths)
    store.hypotheses.write(TOY_RUN, {"toy/c0": "Alpha beta gamma", "toy/c1": "delta epsilon"})
    for normalizer_id in ("leaderboard_split", "own"):
        rows = of_run(ablate(store, [TOY_RUN], normalizer_id, "wer", "overall"))
        # With case no longer forgiven, "Alpha" is not "alpha": one word of five.
        assert rows["without case"]["change"] == pytest.approx(100 * 1 / 5)
        assert rows["without fillers"]["change"] == 0


def test_every_run_has_rows_of_its_own_with_changes_from_its_own_score(paths):
    store = make_run(paths)
    other = Run("other/sys", "toy-bench")
    store.hypotheses.write(other, {"toy/c0": "alpha beta gama", "toy/c1": ""})
    ablate(store, [TOY_RUN], "leaderboard_split", "wer", "overall")  # only the first has its scores saved
    table = ablate(store, [TOY_RUN, other], "leaderboard_split", "wer", "overall")
    assert len(table) == 2 * 14 and set(of_run(table)) == set(of_run(table, other)) == set(ablation.variants(paths, "leaderboard_split"))
    # Four errors in five words for the first, three for the other: each run's changes are from its own figure.
    assert of_run(table)["all families on"]["overall"] == pytest.approx(80.0)
    assert of_run(table, other)["all families on"]["overall"] == pytest.approx(60.0)
    assert of_run(table)["all families on"]["change"] == of_run(table, other)["all families on"]["change"] == 0
    assert store.scores.has(other, "leaderboard_split-without-initialisms", "wer")  # the other's scores are saved now too


def test_a_family_table_worked_out_again_starts_from_the_texts_as_written(paths):
    store = make_run(paths)
    first = ablate(store, [TOY_RUN], "own", "wer", "overall")
    as_normalized = store.hypotheses.read(TOY_RUN, "own-without-case")
    # As if a rule had changed since: what is saved of one variant's texts is no longer what its rules give.
    store.hypotheses.write(TOY_RUN, {"toy/c0": "something", "toy/c1": "else"}, "own-without-case")
    store.references.write("toy-bench", {"toy/c0": "x", "toy/c1": "y", "toy/c2": "z"}, "own-without-case")
    assert ablate(store, [TOY_RUN], "own", "wer", "overall") == first  # the saved scores are read
    assert ablate(store, [TOY_RUN], "own", "wer", "overall", recompute=True) == first
    assert store.hypotheses.read(TOY_RUN, "own-without-case") == as_normalized
    assert store.references.read("toy-bench", "own-without-case")["toy/c0"] == "alpha beta gamma"


def test_the_family_table_is_of_character_errors_too(paths):
    store = make_run(paths)
    assert len(ablate(store, [TOY_RUN], "leaderboard_split", "cer", "overall")) == 14
    assert len(ablate(store, [TOY_RUN], "own", "cer", "overall")) == 18
    with pytest.raises(KeyError, match="overall"):
        ablate(store, [TOY_RUN], "own", "wer", "typical")  # an unknown aggregation names the real ones


class Letters:
    """Stands in for the sentence encoder: a text is how many of each small letter it has."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[text.count(letter) + 0.01 for letter in "abcdefghijklmnopqrstuvwxyz"] for text in texts])


def test_the_family_table_is_of_semantic_error_where_every_variant_has_it(paths):
    store = make_run(paths)
    store.hypotheses.write(TOY_RUN, {"toy/c0": "Alpha beta gamma", "toy/c1": "delta epsilon"})
    other = Run("other/sys", "toy-bench")
    store.hypotheses.write(other, {"toy/c0": "alpha beta gamma", "toy/c1": "delta epsilon"})
    # Semantic error is computed on request, by `ba semantic --families`. Until then the table has no row, and nothing fails.
    assert ablate(store, [TOY_RUN, other], "own", "semantic", "overall") == []
    variants = ablation.variants(paths, "own")
    for normalizer in variants.values():
        scoring.score_semantic(store, [TOY_RUN], normalizer, Letters())
    table = ablate(store, [TOY_RUN, other], "own", "semantic", "overall")
    # The run that has it under every variant has its eighteen rows. The other has none, and is left out.
    assert [row["variant"] for row in table] == list(variants) and {row["run_id"] for row in table} == {TOY_RUN.run_id}
    rows = of_run(table)
    assert (rows["all families on"]["metric"], rows["all families on"]["overall"], rows["all families on"]["change"]) == ("semantic", 0.0, 0)
    # With case no longer forgiven "Alpha" is not "alpha", and the meaning moves a little. No other family changes these texts.
    assert rows["without case"]["change"] > 0
    assert rows["without numerals"]["change"] == 0
    # A run with it under some variants only is left out as well.
    scoring.score_semantic(store, [other], variants["all families on"], Letters())
    assert {row["run_id"] for row in ablate(store, [TOY_RUN, other], "own", "semantic", "overall")} == {TOY_RUN.run_id}
