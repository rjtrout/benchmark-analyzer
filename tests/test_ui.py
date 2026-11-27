"""The web UI, on scratch projects. No network, no account, and no server.

Pages are run with Streamlit's AppTest. The sentence encoder is always a stand-in.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from ba.actions import ablation, scoring, slices
from ba.actions.score_summary import score_table
from ba.metadata.derived import DERIVED
from ba.metadata.given import GIVEN
from ba.metadata.populate import populate_metadata, populate_speaking_rate
from ba.metrics import semantic as semantic_metric
from ba.normalize import leaderboard_split, own
from ba.normalize.registry import normalizers
from ba.paths import Paths
from ba.store.store import Store
from ba.store.test_cases import TestCase
from ba.store.transcriptions import Run
from tests.conftest import REPO_ROOT

APP = REPO_ROOT / "src" / "ba" / "ui" / "app.py"
PAGES = {
    "Results": "pages/results.py",
    "Normalization": "pages/normalization.py",
    "Slices": "pages/slices.py",
    "Benchmark metadata": "pages/metadata.py",
}
SHORT = {"Musci-research/Musci-ASR-qwen-1019-hf": "Musci-ASR-qwen-1019-hf", "salesforce-sovereign-intelligence/SOV-Transcribe-1": "SOV-Transcribe-1"}
# Two of the models in configs/systems.yaml.
FIRST, SECOND = "Musci-research/Musci-ASR-qwen-1019-hf", "salesforce-sovereign-intelligence/SOV-Transcribe-1"
NORMALIZERS = ["verbatim", "case-punct", "whisper-stock", "leaderboard", "leaderboard_split", "own"]

# source id, seconds, reference, what the first model wrote, what the second wrote
TEST_CASES = [
    ("AMI_EN2001a_H03_MEE067_0000100_0000200", 1.0, "alpha beta gamma delta", "alpha bravo gamma delta", "alpha beta gamma delta"),
    ("AMI_EN2001a_H03_MEE067_0000300_0000400", 2.0, "Is it twenty five?", "is it 25", "Is it twenty five?"),
    ("AMI_EN2001a_H01_FEO065_0000500_0000600", 6.0, "epsilon zeta eta", "epsilon zeta", "epsilon zeta eta theta"),
    ("AMI_ES2004a_H02_MED012_0000700_0000800", 9.0, "theta iota kappa lambda", "theta iota kappa lambda mu", "theta iota"),
]
TEST_CASE_IDS = [f"ami_cleaned/{source_id}" for source_id, *_ in TEST_CASES]


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """An empty project that ``Paths.discover()`` finds: the repo's configs copied in, its clone linked."""
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    (tmp_path / "third_party").symlink_to(REPO_ROOT / "third_party", target_is_directory=True)
    monkeypatch.setenv("BA_ROOT", str(tmp_path))
    st.cache_data.clear()
    return tmp_path


def save_list(paths: Paths, benchmark_id: str, test_cases: list[TestCase], references: dict[str, str]) -> Store:
    """A benchmark as a fetch leaves it: its test cases with their metadata, and its references by what is scored."""
    store = Store(paths)
    for test_case in test_cases:
        populate_metadata(test_case, {"audio_length_s": test_case.duration}, references[test_case.unit_id], paths)
    populate_speaking_rate(test_cases)
    store.test_cases.write(benchmark_id, test_cases)
    store.references.write(benchmark_id, references)
    return store


def save_run(store: Store, system_id: str, benchmark_id: str, texts: list[str]) -> Run:
    """A model's transcripts of a benchmark, in list order: one for each test case, or for each session."""
    run = Run(system_id, benchmark_id)
    scored = list(store.references.read(benchmark_id))
    store.hypotheses.write(run, dict(zip(scored, texts, strict=True)))
    return run


def save_earnings(paths: Paths, chunks: int) -> Store:
    """Earnings22's layout: chunks of one call, and the call's one reference."""
    test_cases = [TestCase(f"earnings/{i}", parent_id="call-1", chunk_index=i, duration=30.0) for i in range(chunks)]
    return save_list(paths, "earnings22-cleaned", test_cases, {"earnings/call-1": "alpha beta gamma delta"})


@pytest.fixture
def project(root) -> Paths:
    """Four AMI test cases, with two models' transcripts of them."""
    paths = Paths(root=root)
    test_cases = []
    references = {}
    for test_case_id, (_, seconds, reference, _, _) in zip(TEST_CASE_IDS, TEST_CASES, strict=True):
        test_cases.append(TestCase(test_case_id, duration=seconds))
        references[test_case_id] = reference
    store = save_list(paths, "ami-cleaned", test_cases, references)
    save_run(store, FIRST, "ami-cleaned", [row[3] for row in TEST_CASES])
    save_run(store, SECOND, "ami-cleaned", [row[4] for row in TEST_CASES])
    return paths


def open_page(name: str) -> AppTest:
    """Open the app, as ``streamlit run src/ba/ui/app.py`` does, and go to a page."""
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    return at if name == "Results" else at.switch_page(PAGES[name]).run()


def healthy(at: AppTest) -> AppTest:
    """No traceback and no error box on the page."""
    assert [e.value for e in at.exception] == []
    assert [e.value for e in at.error] == []
    return at


def page_text(at: AppTest) -> str:
    """Everything the page holds, tables and charts included, as text."""
    return "\n".join(str(element.proto) for element in [*at.main, *at.sidebar] if getattr(element, "proto", None) is not None)


def table_with(at: AppTest, *columns: str) -> pd.DataFrame:
    """The first table on the page that has these columns."""
    for element in at.dataframe:
        if set(columns) <= set(element.value.columns):
            return element.value
    raise AssertionError(f"No table with columns {columns}. Tables: {[list(e.value.columns) for e in at.dataframe]}")


def charts(at: AppTest) -> list[dict]:
    """Every chart on the page as its Vega-Lite spec, with the data rows Streamlit sends beside it put back in."""
    specs = []
    for element in at.get("vega_lite_chart"):
        spec = json.loads(element.proto.spec)
        spec["datasets"] = {}
        for dataset in element.proto.datasets:
            sent = pa.ipc.open_stream(dataset.data.data).read_pandas()
            spec["datasets"][dataset.name] = sent.astype(object).where(sent.notna(), None).to_dict("records")
        specs.append(spec)
    return specs


def rows_of(spec: dict) -> list[dict]:
    """A chart's data rows: the dataset its spec names."""
    name = (spec.get("data") or {}).get("name")
    return spec["datasets"].get(name) or next(iter(spec["datasets"].values()), [])


def encodings(spec: dict) -> dict:
    """The encoding of a chart's first layer, through a facet if there is one."""
    inner = spec.get("spec", spec)
    return (inner.get("layer") or [inner])[0]["encoding"]


def expected(paths: Paths, system_id: str, metric: str, agg: str) -> dict[str, float]:
    """A model's figure under each normalizer, worked out by the library the command line uses."""
    store = Store(paths)
    run = Run(system_id, "ami-cleaned")
    table = score_table(store, [run], [normalizers(paths).get(i) for i in NORMALIZERS], [agg], metric_ids=[metric])
    return {row["normalizer"]: row[agg] for row in table}


# ── every page opens ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_opens_on_an_empty_project_and_says_what_to_do(root, page):
    at = healthy(open_page(page))
    said = " ".join(i.value for i in at.info)
    assert "`ba fetch`" in said  # each names the command that gives it something to show


def add_urgent(project: Paths, models=(FIRST, SECOND)) -> Store:
    """A second benchmark, urgent-clean: two test cases, and each of these models' transcripts of them."""
    test_cases = [TestCase(f"urgent2024_clean/u{i}", duration=5.0) for i in range(2)]
    store = save_list(project, "urgent-clean", test_cases, {test_case.test_case_id: "alpha beta gamma" for test_case in test_cases})
    texts = {FIRST: ["alpha beta", "alpha beta gamma"], SECOND: ["alpha beta gamma", "alpha beta gamma"]}
    for model in models:
        save_run(store, model, "urgent-clean", texts[model])
    st.cache_data.clear()
    return store


def figure(paths: Paths, system_id: str, benchmark_id: str, metric: str, normalizer_id: str, agg: str) -> float:
    """One figure, worked out by the library the command line uses."""
    store = Store(paths)
    run = Run(system_id, benchmark_id)
    [row] = score_table(store, [run], [normalizers(paths).get(normalizer_id)], [agg], metric_ids=[metric])
    return row[agg]


# ── Results: the models on the x-axis, a line per metric ─────────────────────────────────


def test_results_put_the_models_on_the_x_axis_with_a_line_per_metric(project):
    at = healthy(open_page("Results"))
    agg, normalizer_id = at.radio(key="results_agg").value, at.selectbox(key="results_normalizer").value  # what the page shows
    assert at.multiselect(key="results_benchmarks").value == ["ami-cleaned"]  # every benchmark that has transcripts
    assert at.pills(key="results_models").value == [FIRST, SECOND]  # the models that have transcripts here
    [chart] = charts(at)  # one benchmark: its own chart, and nothing to combine
    rows = rows_of(chart)
    drawn = {(r["system"], r["metric"]): r["value"] for r in rows}
    # The numbers are the ones the command line prints.
    for system_id in (FIRST, SECOND):
        assert drawn[(system_id, "WER")] == pytest.approx(figure(project, system_id, "ami-cleaned", "wer", normalizer_id, agg))
        assert drawn[(system_id, "CER")] == pytest.approx(figure(project, system_id, "ami-cleaned", "cer", normalizer_id, agg))
    assert {r["metric"] for r in rows} == {"WER", "CER"}  # semantic error is not computed yet: no line is made up for it
    enc = encodings(chart)
    assert enc["x"]["field"] == "model" and enc["x"]["sort"] == [SHORT[FIRST], SHORT[SECOND]]  # the models, in the table's order
    assert {r["model"] for r in rows} == set(SHORT.values())
    assert enc["y"]["field"] == "value" and enc["y"]["title"] == f"Error rate, {agg} (%)"
    assert enc["color"]["field"] == "metric" and enc["color"]["title"] == "Metric"
    table = table_with(at, "model", "benchmark", "WER", "CER").set_index(["model", "benchmark"])
    assert table.loc[(FIRST, "ami-cleaned"), "WER"] == pytest.approx(drawn[(FIRST, "WER")])
    assert table.loc[(FIRST, "ami-cleaned"), "scored / test cases"] == "4/4"


@pytest.mark.parametrize(("agg", "normalizer_id"), [("mean", "verbatim"), ("median", "leaderboard")])
def test_results_follow_the_aggregation_and_the_normalizer(project, agg, normalizer_id):
    at = open_page("Results")
    at.radio(key="results_agg").set_value(agg)
    at.selectbox(key="results_normalizer").set_value(normalizer_id).run()
    [chart] = charts(healthy(at))
    drawn = {(r["system"], r["metric"]): r["value"] for r in rows_of(chart)}
    assert drawn[(FIRST, "WER")] == pytest.approx(figure(project, FIRST, "ami-cleaned", "wer", normalizer_id, agg))
    assert drawn[(SECOND, "CER")] == pytest.approx(figure(project, SECOND, "ami-cleaned", "cer", normalizer_id, agg))
    assert encodings(chart)["y"]["title"] == f"Error rate, {agg} (%)"


def test_several_benchmarks_give_a_large_combined_chart_above_the_individual_ones(project):
    add_urgent(project)
    at = healthy(open_page("Results"))
    assert at.multiselect(key="results_benchmarks").options == ["urgent-clean", "ami-cleaned"]  # the config's order
    assert at.multiselect(key="results_benchmarks").value == ["urgent-clean", "ami-cleaned"]
    combined, individual = charts(at)
    each = {(r["system"], r["benchmark"], r["metric"]): r["value"] for r in rows_of(individual)}
    whole = {(r["system"], r["metric"]): r for r in rows_of(combined)}
    for system_id in (FIRST, SECOND):
        for metric in ("WER", "CER"):
            # Each benchmark counts the same, as in the leaderboard's own average.
            mean = (each[(system_id, "urgent-clean", metric)] + each[(system_id, "ami-cleaned", metric)]) / 2
            assert whole[(system_id, metric)]["value"] == pytest.approx(mean)
            assert whole[(system_id, metric)]["sets"] == 2
    assert each[(FIRST, "urgent-clean", "WER")] == pytest.approx(figure(project, FIRST, "urgent-clean", "wer", "leaderboard", "overall"))
    # The combined chart is one large panel. The individual ones are a panel per benchmark, and smaller.
    assert individual["facet"]["field"] == "benchmark" and len({r["benchmark"] for r in rows_of(individual)}) == 2
    assert len({r["benchmark"] for r in rows_of(combined)}) == 1 and "2 benchmarks" in rows_of(combined)[0]["benchmark"]
    assert combined["spec"]["width"] > individual["spec"]["width"] and combined["spec"]["height"] > individual["spec"]["height"]
    table = table_with(at, "model", "benchmark", "WER").set_index(["model", "benchmark"])
    assert table.loc[(FIRST, "combined"), "WER"] == pytest.approx(whole[(FIRST, "WER")]["value"])
    assert list(table.index.get_level_values("benchmark"))[:2] == ["combined", "combined"]  # the combined score first
    # One benchmark: its own chart alone.
    at.multiselect(key="results_benchmarks").set_value(["ami-cleaned"]).run()
    assert len(charts(healthy(at))) == 1
    at.multiselect(key="results_benchmarks").set_value([]).run()
    assert "Pick at least one benchmark" in " ".join(i.value for i in healthy(at).info)


def test_the_combined_score_is_over_the_benchmarks_every_chosen_model_has(project):
    add_urgent(project, models=(FIRST,))  # the second model has no transcripts of urgent-clean
    at = healthy(open_page("Results"))
    combined, _ = charts(at)
    whole = {(r["system"], r["metric"]): r for r in rows_of(combined)}
    # Both models are compared on the one benchmark both have.
    assert {r["sets"] for r in whole.values()} == {1}
    assert whole[(FIRST, "WER")]["value"] == pytest.approx(figure(project, FIRST, "ami-cleaned", "wer", "leaderboard", "overall"))
    assert "WER, CER: urgent-clean is left out of the combined score, where not every chosen model has a figure." in page_text(at)
    # With the second model switched off, the first is scored on both.
    at.pills(key="results_models").set_value([FIRST]).run()
    combined, _ = charts(healthy(at))
    assert {(r["system"], r["sets"]) for r in rows_of(combined)} == {(FIRST, 2)}


def test_models_can_be_switched_off_on_results(project):
    at = open_page("Results")
    at.pills(key="results_models").set_value([SECOND]).run()
    [chart] = charts(healthy(at))
    assert {r["system"] for r in rows_of(chart)} == {SECOND} and encodings(chart)["x"]["sort"] == [SHORT[SECOND]]
    assert set(table_with(at, "model", "WER")["model"]) == {SECOND}
    at.pills(key="results_models").set_value([]).run()
    assert "Pick at least one model" in " ".join(i.value for i in healthy(at).info) and charts(at) == []


# ── Results: semantic error ──────────────────────────────────────────────────────────────


class WordBag:
    """Stands in for the sentence encoder: a text is the bag of its letters."""

    def __init__(self, model: str = "test/word-bag") -> None:
        self.id = model

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[text.lower().count(letter) for letter in "abcdefghijklmnopqrstuvwxyz"] for text in texts], dtype=float)


def test_semantic_error_says_what_is_not_computed_yet_and_offers_to_compute_it(project):
    at = healthy(open_page("Results"))
    assert "not computed yet for 12 model and normalizer pairs" in " ".join(i.value for i in at.info)  # 2 models x 6 normalizers
    assert "results_semantic" in [b.key for b in at.button] and [w.value for w in at.warning] == []
    [chart] = charts(at)
    assert {r["metric"] for r in rows_of(chart)} == {"WER", "CER"}  # the other two lines are drawn all the same


def test_semantic_error_is_computed_from_the_page_and_then_drawn_as_a_third_line(project, monkeypatch):
    monkeypatch.setattr(semantic_metric, "SentenceTransformerEmbedder", WordBag)
    at = healthy(open_page("Results"))
    at.button(key="results_semantic").click().run()
    healthy(at)
    assert "Semantic error computed for 12" in " ".join(s.value for s in at.success)
    [chart] = charts(at)
    stored = Store(project).scores.read(Run(SECOND, "ami-cleaned"), "leaderboard", "semantic")
    assert len(stored) == 4 and stored[0]["rate"] == 0.0  # the second model's first transcript is the reference
    drawn = {(r["system"], r["metric"]): r["value"] for r in rows_of(chart)}
    assert {metric for _, metric in drawn} == {"WER", "CER", "Semantic error"}
    # Each test case's weight is its reference's word count, which its word errors hold.
    assert drawn[(SECOND, "Semantic error")] == pytest.approx(100 * sum(r["rate"] * r["n_ref"] for r in stored) / sum(r["n_ref"] for r in stored))
    assert encodings(chart)["color"]["sort"] == ["WER", "CER", "Semantic error"]
    # Once stored there is nothing left to compute, so the button is gone.
    assert "results_semantic" not in [b.key for b in at.button]


def test_a_benchmark_with_no_semantic_error_is_left_out_of_that_lines_combined_score(project):
    # Earnings22's layout: scored per session, which a sentence encoder cannot read.
    add_urgent(project)
    store = save_earnings(project, 2)
    for model in (FIRST, SECOND):
        save_run(store, model, "earnings22-cleaned", ["alpha beta gamma delta"])
    for benchmark_id in ("urgent-clean", "ami-cleaned"):
        scoring.score_semantic(store, store.hypotheses.runs([FIRST, SECOND], [benchmark_id]), normalizers(project).get("leaderboard"), WordBag())
    st.cache_data.clear()
    at = healthy(open_page("Results"))
    combined, _ = charts(at)
    sets = {r["metric"]: r["sets"] for r in rows_of(combined)}
    assert sets == {"WER": 3, "CER": 3, "Semantic error": 2}
    assert "Semantic error: earnings22-cleaned is left out of the combined score, where no model has a figure." in page_text(at)


# ── Normalization: the models on the x-axis, a line per normalizer ───────────────────────


def test_normalization_puts_the_models_on_the_x_axis_with_a_line_per_normalizer(project):
    at = open_page("Normalization")
    at.radio(key="normalization_metric").set_value("WER")
    at.radio(key="normalization_agg").set_value("overall").run()
    assert healthy(at).pills(key="normalization_models").value == [FIRST, SECOND]
    chart, _, _ = charts(at)  # the lines, and under them the two heatmaps of the rule families
    rows = rows_of(chart)
    drawn = {(r["system"], r["normalizer"]): r["value"] for r in rows}
    for system_id in (FIRST, SECOND):
        for normalizer_id, value in expected(project, system_id, "wer", "overall").items():
            assert drawn[(system_id, normalizer_id)] == pytest.approx(value)
    enc = encodings(chart)
    assert enc["x"]["field"] == "model" and enc["x"]["sort"] == [SHORT[FIRST], SHORT[SECOND]]
    assert enc["color"]["field"] == "normalizer" and enc["color"]["sort"] == NORMALIZERS  # the normalizers, in the registry's order
    assert enc["y"]["title"] == "WER, overall (%)"
    table = table_with(at, "model", "benchmark", "verbatim", "leaderboard").set_index(["model", "benchmark"])
    assert table.loc[(FIRST, "ami-cleaned"), "leaderboard"] == pytest.approx(expected(project, FIRST, "wer", "overall")["leaderboard"])


@pytest.mark.parametrize(("metric", "metric_id", "agg"), [("CER", "cer", "mean"), ("WER", "wer", "median")])
def test_normalization_follows_the_metric_and_the_aggregation(project, metric, metric_id, agg):
    at = open_page("Normalization")
    at.radio(key="normalization_metric").set_value(metric)
    at.radio(key="normalization_agg").set_value(agg).run()
    chart, _, _ = charts(healthy(at))
    drawn = {(r["system"], r["normalizer"]): r["value"] for r in rows_of(chart)}
    for normalizer_id, value in expected(project, FIRST, metric_id, agg).items():
        assert drawn[(FIRST, normalizer_id)] == pytest.approx(value)
    assert encodings(chart)["y"]["title"] == f"{metric}, {agg} (%)"


def test_normalization_combines_benchmarks_and_lets_models_be_switched_off(project):
    add_urgent(project)
    at = healthy(open_page("Normalization"))
    combined, individual, _, _ = charts(at)
    each = {(r["system"], r["benchmark"], r["normalizer"]): r["value"] for r in rows_of(individual)}
    whole = {(r["system"], r["normalizer"]): r["value"] for r in rows_of(combined)}
    assert whole[(FIRST, "verbatim")] == pytest.approx((each[(FIRST, "urgent-clean", "verbatim")] + each[(FIRST, "ami-cleaned", "verbatim")]) / 2)
    assert combined["spec"]["width"] > individual["spec"]["width"]
    at.pills(key="normalization_models").set_value([FIRST]).run()
    combined, individual, _, _ = charts(healthy(at))
    assert {r["system"] for r in [*rows_of(combined), *rows_of(individual)]} == {FIRST}


def test_normalization_needs_semantic_error_computed_first(project, monkeypatch):
    monkeypatch.setattr(semantic_metric, "SentenceTransformerEmbedder", WordBag)
    at = open_page("Normalization")
    at.radio(key="normalization_metric").set_value("Semantic error").run()
    assert charts(healthy(at)) == []  # nothing is made up before it is computed
    at.button(key="normalization_semantic").click().run()
    [chart] = charts(healthy(at))
    assert {r["normalizer"] for r in rows_of(chart)} == set(NORMALIZERS) and encodings(chart)["y"]["title"] == "Semantic error, overall (%)"


# ── Normalization: what each rule family helps ───────────────────────────────────────────


def score_files(project: Paths) -> dict[str, int]:
    """Every file of scores worked out here, with when it was last written."""
    folder = project.artifacts / "scores" / "ours"
    return {str(file.relative_to(folder)): file.stat().st_mtime_ns for file in folder.glob("*/*.parquet")}


def test_the_family_heatmap_shows_what_each_family_is_worth(project):
    at = healthy(open_page("Normalization"))
    lines, heatmap, own_heatmap = charts(at)  # they are there as the page opens: nothing waits to be asked for
    first = Run(FIRST, "ami-cleaned")
    worked = ablation.ablate(Store(project), [first], "leaderboard_split", "wer", "overall")
    drawn = {(r["system"], r["family"]): r["change"] for r in rows_of(heatmap)}
    for row in worked[1:]:
        assert drawn[(FIRST, row["variant"].removeprefix("without "))] == pytest.approx(row["change"])
    # With numbers left as written, "twenty five" against "25" costs the first model two more errors in one more word.
    assert drawn[(FIRST, "numerals")] == pytest.approx(100 * 5 / 15 - 100 * 3 / 14)
    # The second model wrote the number as the reference has it. Its three errors elsewhere are then over one more word.
    assert drawn[(SECOND, "numerals")] == pytest.approx(100 * 3 / 15 - 100 * 3 / 14)
    assert {r["family"] for r in rows_of(heatmap)} == set(leaderboard_split.FAMILIES)


def test_a_second_heatmap_under_the_first_is_of_our_own_normalizer(project):
    at = healthy(open_page("Normalization"))
    _, heatmap, own_heatmap = charts(at)
    # The leaderboard's thirteen families first, and under it our own seventeen.
    assert {r["family"] for r in rows_of(heatmap)} == set(leaderboard_split.FAMILIES)
    assert {r["family"] for r in rows_of(own_heatmap)} == set(own.FAMILIES)
    assert encodings(own_heatmap)["x"]["sort"] == list(own.FAMILIES)
    first = Run(FIRST, "ami-cleaned")
    worked = ablation.ablate(Store(project), [first], "own", "wer", "overall")
    drawn = {(r["system"], r["family"]): r["change"] for r in rows_of(own_heatmap)}
    for row in worked[1:]:
        assert drawn[(FIRST, row["variant"].removeprefix("without "))] == pytest.approx(row["change"])
    text = page_text(at)
    assert "`leaderboard_split`" in text and "`own`" in text


@pytest.mark.parametrize(
    ("urgent_models", "shared_benchmarks"),
    [((FIRST, SECOND), ("ami-cleaned", "urgent-clean")), ((FIRST,), ("ami-cleaned",))],
)
def test_the_family_heatmap_can_combine_benchmarks(project, urgent_models, shared_benchmarks):
    add_urgent(project, models=urgent_models)
    at = healthy(open_page("Normalization"))
    separate = rows_of(charts(at)[-2])  # the last two charts are the heatmaps: the leaderboard's rules, then our own
    assert len({row["row"] for row in separate}) == 2 + len(urgent_models)

    at.toggle(key="normalization_combine_benchmarks").set_value(True).run()
    combined = rows_of(charts(healthy(at))[-2])
    assert {row["row"] for row in combined} == {FIRST, SECOND}
    assert len(combined) == 2 * len(leaderboard_split.FAMILIES)
    # The one switch combines both heatmaps.
    assert len(rows_of(charts(at)[-1])) == 2 * len(own.FAMILIES)

    for row in combined:
        model, family = row["system"], row["family"]
        matching = [
            before for before in separate
            if before["system"] == model and before["family"] == family and before["benchmark"] in shared_benchmarks
        ]
        assert row["sets"] == len(matching) == len(shared_benchmarks)
        assert row["change"] == pytest.approx(np.mean([before["change"] for before in matching]))
        baseline = np.mean([before["with every family on"] for before in matching])
        assert row["with every family on"] == pytest.approx(baseline)

    at.toggle(key="normalization_combine_benchmarks").set_value(False).run()
    assert rows_of(charts(healthy(at))[-2]) == separate


def test_the_family_heatmap_is_read_from_the_scores_once_they_are_saved(project):
    at = healthy(open_page("Normalization"))
    store = Store(project)
    first, second = Run(FIRST, "ami-cleaned"), Run(SECOND, "ami-cleaned")
    # A family switched off is a normalizer like any other: each model has a file of scores under each.
    written = score_files(project)
    assert f"{first.run_id}/leaderboard_split-without-numerals.parquet" in written and f"{second.run_id}/leaderboard_split-without-initialisms.parquet" in written
    assert f"{first.run_id}/own-without-numerals.parquet" in written and f"{second.run_id}/own-without-repetitions.parquet" in written
    # Another aggregation is read from the same scores.
    at.radio(key="normalization_agg").set_value("mean").run()
    _, heatmap, _ = charts(healthy(at))
    mean = {row["variant"]: row["change"] for row in ablation.ablate(store, [first], "leaderboard_split", "wer", "mean")}
    drawn = {(r["system"], r["family"]): r["change"] for r in rows_of(heatmap)}
    assert drawn[(FIRST, "numerals")] == pytest.approx(mean["without numerals"])
    # A new visit, after the server has forgotten everything, reads them too.
    st.cache_data.clear()
    at = healthy(open_page("Normalization"))
    lines, heatmap, own_heatmap = charts(at)
    assert {r["system"] for r in rows_of(heatmap)} == {r["system"] for r in rows_of(own_heatmap)} == {FIRST, SECOND}
    # They follow the models chosen.
    at.pills(key="normalization_models").set_value([SECOND]).run()
    _, heatmap, own_heatmap = charts(healthy(at))
    assert {r["system"] for r in rows_of(heatmap)} == {r["system"] for r in rows_of(own_heatmap)} == {SECOND}
    assert score_files(project) == written  # nothing was scored again along the way


def test_the_family_heatmap_scores_only_what_is_not_scored_yet(project):
    store = Store(project)
    first, second = Run(FIRST, "ami-cleaned"), Run(SECOND, "ami-cleaned")
    ablation.ablate(store, [first], "own", "wer", "overall")  # as `ba ablate --system <first>` leaves it
    # The first model's scores under each of own's families switched off: seventeen files.
    before = {name: stamp for name, stamp in score_files(project).items() if name.startswith(f"{first.run_id}/own-")}
    assert len(before) == 17 and not store.scores.has(second, "own-without-numerals", "wer")
    assert not store.scores.has(first, "leaderboard_split-without-numerals", "wer")
    at = healthy(open_page("Normalization"))
    _, heatmap, own_heatmap = charts(at)
    assert {r["system"] for r in rows_of(heatmap)} == {r["system"] for r in rows_of(own_heatmap)} == {FIRST, SECOND}
    after = score_files(project)
    assert {name: after[name] for name in before} == before  # the first's scores were read, not counted again
    assert store.scores.has(second, "own-without-numerals", "wer") and store.scores.has(first, "leaderboard_split-without-numerals", "wer")


def test_the_family_heatmaps_draw_semantic_error_once_it_is_computed_for_the_families(project):
    at = open_page("Normalization")
    at.radio(key="normalization_metric").set_value("Semantic error").run()
    # Not computed for the families yet: the page says how to, and draws no heatmap.
    assert "ba semantic --families" in page_text(healthy(at)) and charts(at) == []

    store = Store(project)
    runs = [Run(FIRST, "ami-cleaned"), Run(SECOND, "ami-cleaned")]
    for normalizer_id in ("leaderboard_split", "own"):
        for normalizer in ablation.variants(project, normalizer_id).values():
            scoring.score_semantic(store, runs, normalizer, WordBag())
    st.cache_data.clear()
    at = open_page("Normalization")
    at.radio(key="normalization_metric").set_value("Semantic error").run()
    lines, heatmap, own_heatmap = charts(healthy(at))
    assert {r["family"] for r in rows_of(heatmap)} == set(leaderboard_split.FAMILIES)
    assert {r["family"] for r in rows_of(own_heatmap)} == set(own.FAMILIES)
    worked = ablation.ablate(store, [runs[0]], "own", "semantic", "overall")
    drawn = {(r["system"], r["family"]): r["change"] for r in rows_of(own_heatmap)}
    for row in worked[1:]:
        assert drawn[(FIRST, row["variant"].removeprefix("without "))] == pytest.approx(row["change"])
    assert "ba semantic --families" not in page_text(at)


# ── Slices ───────────────────────────────────────────────────────────────────────────────

def test_slices_show_two_premade_slices_side_by_side(project):
    at = healthy(open_page("Slices"))
    left, right = at.selectbox(key="slices_left_by"), at.selectbox(key="slices_right_by")
    # Each side offers the premade slices, and its chart is of the one chosen there.
    assert left.options == right.options == list(slices.SLICES.values())
    first, second = charts(at)
    assert (encodings(first)["x"]["title"], encodings(second)["x"]["title"]) == (slices.SLICES[left.value], slices.SLICES[right.value])
    assert {r["benchmark"] for r in rows_of(first)} == {r["benchmark"] for r in rows_of(second)} == {"ami-cleaned"}
    # One legend for the two charts, above them: each model's colour beside its name. A legend under each
    # chart would take the room the lines need.
    colour = encodings(first)["color"]
    assert colour["legend"] is None and encodings(second)["color"]["scale"] == colour["scale"]
    assert all(f"background:{shade}" in page_text(at) for shade in colour["scale"]["range"])


def test_slices_draw_a_numbers_bins_where_their_test_cases_are_with_a_line_per_model(project):
    at = healthy(open_page("Slices"))
    at.number_input(key="slices_bins").set_value(2).run()
    chart, _ = charts(healthy(at))
    rows = rows_of(chart)
    assert {r["system"] for r in rows} == {FIRST, SECOND}
    low, high = sorted((r for r in rows if r["system"] == FIRST), key=lambda r: r["at"])
    # A bin is drawn at the mean of its test cases' values, so the higher values are further right.
    assert (low["at"], high["at"]) == (1.5, 7.5) and (low["test_cases"], high["test_cases"]) == (2, 2)
    assert low["value"].startswith("1.00 to") and high["value"].endswith("9.00")
    # The two short test cases: one error in seven reference words for the first model, under the leaderboard's rules.
    assert low["score"] == pytest.approx(100 * 1 / 7) and high["score"] == pytest.approx(100 * 2 / 7)
    enc = encodings(chart)
    assert (enc["x"]["field"], enc["x"]["type"]) == ("at", "quantitative")
    assert enc["color"]["field"] == "system" and enc["y"]["title"] == "WER, overall (%)"
    table = table_with(at, "model", "Duration (s)", "test cases")
    assert len(table) == 4 and set(table["test cases"]) == {2}


def test_each_side_of_slices_has_a_benchmark_of_its_own(project):
    test_cases = [TestCase(f"urgent2024_clean/u{i}", duration=5.0 + i) for i in range(3)]
    store = save_list(project, "urgent-clean", test_cases, {test_case.test_case_id: "alpha beta gamma" for test_case in test_cases})
    save_run(store, FIRST, "urgent-clean", ["alpha beta gamma", "alpha beta", "alpha"])
    st.cache_data.clear()
    at = healthy(open_page("Slices"))
    assert at.selectbox(key="slices_left_benchmark").options == ["urgent-clean", "ami-cleaned"]
    at.selectbox(key="slices_left_benchmark").set_value("ami-cleaned")
    at.selectbox(key="slices_right_benchmark").set_value("urgent-clean")
    at.number_input(key="slices_bins").set_value(10)  # more bins than a set here has lengths: a point for each
    at.selectbox(key="slices_right_by").set_value("duration").run()
    left, right = charts(healthy(at))
    # The same slice of two benchmarks, side by side.
    assert {r["benchmark"] for r in rows_of(left)} == {"ami-cleaned"} and {r["benchmark"] for r in rows_of(right)} == {"urgent-clean"}
    assert [(r["at"], r["score"]) for r in sorted(rows_of(right), key=lambda r: r["at"])] == [(5.0, 0.0), (6.0, pytest.approx(100 / 3)), (7.0, pytest.approx(200 / 3))]


def test_models_can_be_switched_off_on_slices_and_keep_their_colours(project):
    at = healthy(open_page("Slices"))
    assert at.pills(key="slices_models").value == [FIRST, SECOND]
    both, _ = charts(at)
    colours = dict(zip(encodings(both)["color"]["scale"]["domain"], encodings(both)["color"]["scale"]["range"], strict=True))
    at.pills(key="slices_models").set_value([SECOND]).run()
    for chart in charts(healthy(at)):
        assert {r["system"] for r in rows_of(chart)} == {SECOND}
        # A model keeps its colour when another is switched off.
        assert encodings(chart)["color"]["scale"] == {"domain": [SECOND], "range": [colours[SECOND]]}
    at.pills(key="slices_models").set_value([]).run()
    assert "Pick at least one model" in " ".join(i.value for i in healthy(at).info)


def test_slices_draw_a_flag_as_false_and_true_and_follow_the_metric_and_aggregation(project):
    at = open_page("Slices")
    at.radio(key="slices_metric").set_value("CER")
    at.radio(key="slices_agg").set_value("mean")
    at.selectbox(key="slices_left_by").set_value("alphanumeric").run()
    chart, _ = charts(healthy(at))
    assert {r["value"] for r in rows_of(chart)} == {"false", "true"}
    enc = encodings(chart)
    assert (enc["x"]["field"], enc["x"]["type"], enc["x"]["title"]) == ("value", "nominal", "Alphanumeric present")
    assert enc["y"]["title"] == "CER, mean (%)"


def test_slices_say_how_to_get_a_field_measured_on_the_audio(project):
    at = open_page("Slices")
    at.selectbox(key="slices_left_by").set_value("level").run()
    said = " ".join(i.value for i in healthy(at).info)
    assert "No test case of ami-cleaned has a value for Level (dBFS)." in said and "ba fetch --audio --benchmark ami-cleaned" in said
    [other] = charts(at)  # the other side is drawn all the same
    assert encodings(other)["x"]["title"] == "Reference words"


def test_slices_say_when_no_test_case_of_the_benchmark_has_a_value(project):
    at = open_page("Slices")
    at.selectbox(key="slices_left_by").set_value("snr_given").run()
    assert "No test case of ami-cleaned has a value for SNR, given (dB)." in " ".join(i.value for i in healthy(at).info)
    assert len(charts(at)) == 1


def test_slices_leave_out_a_set_scored_per_session_and_say_why(project):
    store = save_earnings(project, 2)
    save_run(store, FIRST, "earnings22-cleaned", ["alpha beta gamma delta"])
    st.cache_data.clear()
    at = healthy(open_page("Slices"))
    assert "Left out: earnings22-cleaned" in page_text(at)
    assert at.selectbox(key="slices_left_benchmark").options == at.selectbox(key="slices_right_benchmark").options == ["ami-cleaned"]
    assert all({r["benchmark"] for r in rows_of(chart)} == {"ami-cleaned"} for chart in charts(at))


def test_slices_need_semantic_error_computed_first_and_then_break_it_down(project, monkeypatch):
    at = open_page("Slices")
    at.radio(key="slices_metric").set_value("Semantic error").run()
    assert "Compute it on the Results page" in " ".join(i.value for i in healthy(at).info) and charts(at) == []
    store = Store(project)
    normalizer = normalizers(project).get("leaderboard")
    scoring.score_semantic(store, store.hypotheses.runs([FIRST, SECOND], ["ami-cleaned"]), normalizer, WordBag())
    st.cache_data.clear()
    at = open_page("Slices")
    at.number_input(key="slices_bins").set_value(10)
    at.radio(key="slices_metric").set_value("Semantic error").run()
    chart, _ = charts(healthy(at))
    assert encodings(chart)["y"]["title"] == "Semantic error, overall (%)" and len(rows_of(chart)) == 8  # 2 models x 4 lengths


# ── Slices: the examples under a chart ───────────────────────────────────────────────────


def many(project: Paths, count: int = 23) -> Store:
    """A second benchmark, every test case five seconds long. The first model drops 0 to 4 of each reference's four words."""
    test_cases = [TestCase(f"urgent2024_clean/u{i:02d}", duration=5.0) for i in range(count)]
    store = save_list(project, "urgent-clean", test_cases, {test_case.test_case_id: "alpha beta gamma delta" for test_case in test_cases})
    words = ["alpha", "beta", "gamma", "delta"]
    save_run(store, FIRST, "urgent-clean", [" ".join(words[: 4 - i % 5]) for i in range(count)])
    st.cache_data.clear()
    return store


def page_key(side: str, benchmark_id: str, by: str, system_id: str, value: str, order: str) -> str:
    """The key of an examples panel's page number: it belongs to what the panel shows."""
    return f"slices_{side}_examples_page_{benchmark_id}_{by}_{system_id}_{value}_{order}"


def test_slices_show_examples_of_a_bin_a_page_at_a_time_with_the_worst_first(project):
    many(project)
    at = open_page("Slices")
    at.selectbox(key="slices_left_benchmark").set_value("urgent-clean").run()
    healthy(at)
    # Under the chart: one model's test cases of one bin. All 23 are five seconds long, so there is one bin.
    assert at.selectbox(key="slices_left_examples_model").value == FIRST
    bucket = at.selectbox(key="slices_left_examples_value")
    assert (bucket.label, bucket.options, bucket.value) == ("Duration (s)", ["5"], "5")
    at.radio(key="slices_left_examples_order").set_value("Highest first").run()
    # The 23 test cases, worst first: the model wrote nothing for four of them, one word of four for the next four, and so on.
    every = ["100.00"] * 4 + ["75.00"] * 4 + ["50.00"] * 5 + ["25.00"] * 5 + ["0.00"] * 5
    written = [""] * 4 + ["alpha"] * 4 + ["alpha beta"] * 5 + ["alpha beta gamma"] * 5 + ["alpha beta gamma delta"] * 5
    to_a_page = slices.EXAMPLES
    pages = math.ceil(len(every) / to_a_page)
    first_page = healthy(at).table[0].value
    assert list(first_page.columns) == ["WER (%)", "reference", "hypothesis"]
    assert list(first_page["WER (%)"]) == every[:to_a_page]
    # The reference and what the model wrote for it.
    assert set(first_page["reference"]) == {"alpha beta gamma delta"}
    assert list(first_page["hypothesis"]) == written[:to_a_page]
    assert f"1 to {min(to_a_page, 23)} of 23 test cases" in page_text(at)
    page = at.number_input(key=page_key("left", "urgent-clean", "duration", FIRST, "5", "Highest first"))
    assert (page.min, page.max, page.value) == (1, pages, 1)
    page.set_value(pages).run()
    last_page = healthy(at).table[0].value
    assert list(last_page["WER (%)"]) == every[to_a_page * (pages - 1) :]
    assert f"{to_a_page * (pages - 1) + 1} to 23 of 23 test cases" in page_text(at)


def test_examples_follow_the_bucket_the_order_and_the_model(project):
    at = open_page("Slices")
    at.number_input(key="slices_bins").set_value(2).run()
    at.radio(key="slices_left_examples_order").set_value("Highest first").run()
    other_side = (at.selectbox(key="slices_right_examples_model").value, at.radio(key="slices_right_examples_order").value)
    low, high = healthy(at).selectbox(key="slices_left_examples_value").options
    assert low.startswith("1.00 to") and high.endswith("9.00")
    # The lower bin, the first model: one word of four is wrong in one test case, and "twenty five" is 25 in the other.
    examples = at.table[0].value
    assert list(examples["WER (%)"]) == ["25.00", "0.00"]
    assert list(examples["reference"]) == ["alpha beta gamma delta", "is it 25"] and list(examples["hypothesis"]) == ["alpha bravo gamma delta", "is it 25"]
    at.selectbox(key="slices_left_examples_value").set_value(high).run()
    assert list(healthy(at).table[0].value["WER (%)"]) == ["33.33", "25.00"]
    at.radio(key="slices_left_examples_order").set_value("Lowest first").run()
    assert list(healthy(at).table[0].value["WER (%)"]) == ["25.00", "33.33"]
    at.selectbox(key="slices_left_examples_model").set_value(SECOND).run()
    examples = healthy(at).table[0].value
    assert list(examples["WER (%)"]) == ["33.33", "50.00"] and list(examples["hypothesis"]) == ["epsilon zeta eta theta", "theta iota"]
    # The other side has examples of its own, untouched by these choices.
    assert (at.selectbox(key="slices_right_examples_model").value, at.radio(key="slices_right_examples_order").value) == other_side
    assert len(at.table) == 2


def test_examples_show_the_texts_as_the_chosen_normalizer_leaves_them(project):
    at = open_page("Slices")
    at.number_input(key="slices_bins").set_value(10)  # more bins than there are lengths: each length is a bin
    at.selectbox(key="slices_normalizer").set_value("verbatim").run()
    at.selectbox(key="slices_left_examples_value").set_value("2").run()
    [example] = healthy(at).table[0].value.to_dict("records")
    # As written, case and the question mark count, and so does a number spelt out: only "it" is the same.
    assert (example["reference"], example["hypothesis"], example["WER (%)"]) == ("Is it twenty five?", "is it 25", "75.00")
    assert "as the verbatim normalizer leaves them" in page_text(at)


def test_examples_are_of_a_model_that_is_switched_on(project):
    at = open_page("Slices")
    at.pills(key="slices_models").set_value([SECOND]).run()
    model = healthy(at).selectbox(key="slices_left_examples_model")
    assert model.options == [SHORT[SECOND]] and model.value == SECOND


# ── Benchmark metadata ───────────────────────────────────────────────────────────────────


def titles(at: AppTest) -> list[str]:
    """The field each histogram on the page is of, in page order."""
    return [m.value.split("**")[1] for m in at.markdown if m.value.startswith("**")]


def test_metadata_shows_a_histogram_per_field_for_the_chosen_benchmarks(project):
    at = open_page("Benchmark metadata")
    at.number_input(key="metadata_bins").set_value(20).run()  # enough bars to part four lengths from 1 to 9 seconds
    assert healthy(at).multiselect(key="metadata_benchmarks").value == ["ami-cleaned"]
    shown = titles(at)
    # Every field has a panel. They are laid out in two columns, so the page's order is not the list's.
    assert sorted(shown) == sorted([*GIVEN, *DERIVED])
    by_title = {}
    for spec in charts(at):
        field = spec["encoding"]["x"].get("title")
        by_title[field] = spec
    # A number is counted into bars here, not in the browser: a chart carries a row per bar, however many test cases there are.
    duration = by_title["duration"]
    bars = sorted(rows_of(duration), key=lambda r: r["start"])
    assert [(r["test_cases"], r["share"]) for r in bars] == [(1, 25.0)] * 4 and all(r["start"] < r["end"] for r in bars)
    assert bars[0]["start"] == 1.0 and bars[-1]["end"] == 9.0  # bars of equal width across the range
    assert duration["encoding"]["x"]["bin"] == "binned" and duration["encoding"]["y"]["field"] == "share"
    # A flag is counted per value: one reference of the four holds a number. The bar is the share of the
    # benchmark's test cases, so sets of any size compare.
    flag = by_title["alphanumeric"]
    assert {r["value"]: (r["test_cases"], r["share"]) for r in rows_of(flag)} == {"false": (3, 75.0), "true": (1, 25.0)}
    assert flag["encoding"]["color"]["field"] == "benchmark" and flag["encoding"]["y"]["field"] == "share"
    # A field measured on the audio has no values until the audio is fetched, and says so.
    assert "level" not in by_title and "which `ba fetch --audio` fetches" in page_text(at)
    assert "4 of 4 test cases have a value" in page_text(at)


def test_metadata_can_show_given_fields_or_derived_ones(project):
    at = open_page("Benchmark metadata")
    at.multiselect(key="metadata_sources").set_value(["Given"]).run()
    given = titles(healthy(at))
    assert set(given) == set(GIVEN)
    at.multiselect(key="metadata_sources").set_value(["Derived"]).run()
    derived = titles(healthy(at))
    assert set(derived) == set(DERIVED)
    at.multiselect(key="metadata_sources").set_value([]).run()
    assert "Pick at least one benchmark and one kind of field" in " ".join(i.value for i in healthy(at).info)


def test_metadata_puts_the_chosen_benchmarks_side_by_side(project):
    test_cases = [TestCase(f"urgent2024_clean/u{i}", duration=5.0 + i) for i in range(3)]
    save_list(project, "urgent-clean", test_cases, {test_case.test_case_id: "alpha beta gamma" for test_case in test_cases})  # a list with no run
    st.cache_data.clear()
    at = open_page("Benchmark metadata")
    at.number_input(key="metadata_bins").set_value(20).run()
    assert healthy(at).multiselect(key="metadata_benchmarks").options == ["urgent-clean", "ami-cleaned"]
    durations = next(spec for spec in charts(at) if spec["encoding"]["x"].get("title") == "duration")
    # The two share their bars' edges, and each bar is a share of its own benchmark's test cases: 1 of 4, and 1 of 3.
    drawn = sorted((r["benchmark"], r["start"], r["test_cases"], round(r["share"], 1)) for r in rows_of(durations))
    assert [(b, n, share) for b, _, n, share in drawn] == [("ami-cleaned", 1, 25.0)] * 4 + [("urgent-clean", 1, 33.3)] * 3
    assert len({round(r["end"] - r["start"], 6) for r in rows_of(durations)}) == 1  # bars of one width
    assert "urgent-clean 3 test cases" in page_text(at) and "ami-cleaned 4 test cases" in page_text(at)  # the page's one legend
    # A benchmark keeps its colour from chart to chart.
    scale = durations["encoding"]["color"]["scale"]
    assert scale["domain"] == ["urgent-clean", "ami-cleaned"] and len(set(scale["range"])) == 2
    at.multiselect(key="metadata_benchmarks").set_value(["urgent-clean"]).run()
    durations = next(spec for spec in charts(healthy(at)) if spec["encoding"]["x"].get("title") == "duration")
    assert {r["benchmark"] for r in rows_of(durations)} == {"urgent-clean"}


def test_metadata_counts_a_field_of_the_reference_once_per_session_on_a_set_scored_per_session(project):
    save_earnings(project, 3)  # three chunks of one call
    st.cache_data.clear()
    at = open_page("Benchmark metadata")
    at.multiselect(key="metadata_benchmarks").set_value(["earnings22-cleaned"]).run()
    by_title = {spec["encoding"]["x"].get("title"): spec for spec in charts(healthy(at))}
    assert [(r["test_cases"], r["share"]) for r in rows_of(by_title["ref_words"])] == [(1, 100.0)]  # one call, not three chunks of it
    assert [(r["test_cases"], r["share"]) for r in rows_of(by_title["speaking_rate"])] == [(1, 100.0)]  # and the call's one rate
    assert [(r["test_cases"], r["share"]) for r in rows_of(by_title["duration"])] == [(3, 100.0)]  # a chunk's own length stays per chunk
