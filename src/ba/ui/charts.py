"""The charts the pages draw, as Altair specs. Each is a function of plain rows.

Streamlit applies its own light or dark theme to them, so nothing here names a colour of
the page.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from ba.utils.list_utils import unique

Rows = Sequence[dict[str, Any]]

# Colours for a chart's lines: far apart in hue, and readable on a light page and on a dark one.
MODEL_COLOURS = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#F0E442", "#0072B2"]
# Colours for benchmarks. After twelve they repeat.
BENCHMARK_COLOURS = [
    "#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b",
    "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#17becf", "#bcbd22",
]  # fmt: skip

PANEL_WIDTH = 380
# The combined score, or a benchmark shown alone, gets one large panel.
LARGE_WIDTH = 760
LARGE_HEIGHT = 340
SMALL_HEIGHT = 240


class NoData(ValueError):
    """There is nothing to draw. The message says what would give the chart something."""


# ── names and colours ────────────────────────────────────────────────────────────────────


def _a_colour_each(names: Sequence[str], palette: list[str]) -> dict[str, str]:
    """A colour for each name, in the palette's order. After its last colour the palette starts again."""
    colours = {}
    for index, name in enumerate(names):
        colours[name] = palette[index % len(palette)]
    return colours


def benchmark_colours(benchmark_ids: Sequence[str]) -> dict[str, str]:
    """One colour per benchmark, the same on every chart of a page."""
    return _a_colour_each(benchmark_ids, BENCHMARK_COLOURS)


def model_colours(model_ids: Sequence[str]) -> dict[str, str]:
    """One colour per model, the same whichever models are switched on."""
    return _a_colour_each(model_ids, MODEL_COLOURS)


def short_names(model_ids: Sequence[str]) -> dict[str, str]:
    """Each model's name without its organisation, where that still tells the models apart."""
    short = {}
    for model_id in model_ids:
        short[model_id] = model_id.split("/", 1)[-1]
    if len(set(short.values())) == len(short):
        return short
    # Two models have one name: every model keeps its whole id.
    return {model_id: model_id for model_id in model_ids}


# ── the error rate against the models ────────────────────────────────────────────────────


def model_lines(
    rows: Rows,
    series: str,
    series_title: str,
    series_order: Sequence[str],
    models: Sequence[str],
    labels: dict[str, str],
    y_title: str,
    *,
    large: bool = False,
) -> alt.Chart:
    """The error rate against the models, with a line per series: a metric each, or a normalizer each.

    The models are on the x-axis in the order given, under the names in ``labels``. There
    is one panel per benchmark: small ones side by side, or with ``large`` a big one, for
    the combined score or for a benchmark shown alone. A series keeps its colour whichever
    of them have a figure to draw.
    """
    data = pd.DataFrame([row for row in rows if row.get("value") is not None])
    if data.empty:
        raise NoData("No score to draw yet for this choice.")
    data["model"] = data["system"].map(labels)

    colours = _a_colour_each(series_order, MODEL_COLOURS)
    drawn = set(data[series])
    present = [name for name in series_order if name in drawn]
    panels = unique(data["benchmark"])

    # Every model is named: the labels are slanted, and none is dropped for running into its neighbour.
    x = alt.X(
        "model:N",
        title="Model",
        sort=[labels[model] for model in models],
        axis=alt.Axis(labelAngle=-20, labelLimit=260, labelOverlap=False),
    )
    y = alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=True))
    colour = alt.Color(
        f"{series}:N",
        title=series_title,
        sort=present,
        scale=alt.Scale(domain=present, range=[colours[name] for name in present]),
        legend=alt.Legend(orient="bottom"),
    )
    tooltip = [
        alt.Tooltip("system:N", title="Model"),
        alt.Tooltip("benchmark:N"),
        alt.Tooltip(f"{series}:N", title=series_title),
        alt.Tooltip("value:Q", title=y_title, format=".2f"),
    ]
    if "sets" in data.columns:
        tooltip.append(alt.Tooltip("sets:Q", title="Benchmarks in the mean"))
    base = alt.Chart(data).encode(x=x, y=y, color=colour, tooltip=tooltip)

    if large:
        width, height, columns = LARGE_WIDTH, LARGE_HEIGHT, 1
    else:
        width, height, columns = PANEL_WIDTH, SMALL_HEIGHT, 2
    lines_and_points = alt.layer(base.mark_line(), base.mark_point(filled=True, size=70))
    # Always a facet, even for one panel: its width and height are then those of the plot
    # area, whatever the axis labels and the legend take beside it.
    panel = lines_and_points.properties(width=width, height=height)
    return panel.facet(facet=alt.Facet("benchmark:N", title=None, sort=panels), columns=columns)


# ── a score against a field's values ─────────────────────────────────────────────────────


def slice_lines(rows: Rows, y_title: str, x_title: str, colours: dict[str, str], number: bool) -> alt.Chart:
    """The score against a field's bins or values, with a line per model.

    A ``number`` has its bins on a number line, each where its test cases are (``at``), so the
    higher values are further right. A flag has false, then true. ``colours`` gives each
    model its colour, whichever are drawn. The chart is as wide as the column it is put in,
    and has no legend: two of them share one, which the page draws.
    """
    data = pd.DataFrame(list(rows))
    if data.empty:
        raise NoData("No test case here has a value for this field.")
    drawn = set(data["system"])
    models = [model for model in colours if model in drawn]

    if number:
        x = alt.X("at:Q", title=x_title, scale=alt.Scale(zero=False))
    else:
        x = alt.X("value:N", title=x_title, sort=["false", "true"], axis=alt.Axis(labelAngle=0))
    y = alt.Y("score:Q", title=y_title, scale=alt.Scale(zero=True))
    colour = alt.Color(
        "system:N",
        sort=models,
        scale=alt.Scale(domain=models, range=[colours[model] for model in models]),
        legend=None,
    )
    tooltip = [
        alt.Tooltip("system:N", title="Model"),
        alt.Tooltip("value:N", title=x_title),
        alt.Tooltip("score:Q", title=y_title, format=".2f"),
        alt.Tooltip("test_cases:Q", title="Test cases", format=","),
    ]
    base = alt.Chart(data).encode(x=x, y=y, color=colour, tooltip=tooltip)

    lines_and_points = alt.layer(base.mark_line(), base.mark_point(filled=True, size=40))
    return lines_and_points.properties(width="container", height=LARGE_HEIGHT)


# ── what each rule family is worth ───────────────────────────────────────────────────────


def family_heatmap(rows: Rows, families: Sequence[str], dark: bool = False, *, combined: bool = False) -> alt.Chart:
    """What each rule family of a normalizer is worth, by model and optionally benchmark.

    A cell is the rise in the error rate, in points, when that family is switched off. The
    redder the cell, the more the family helps. ``dark`` says the page is drawn dark, so the
    numbers on the faint cells are written light.
    """
    data = pd.DataFrame(list(rows))
    if data.empty:
        raise NoData("Nothing to take apart yet.")
    data["row"] = data["system"] if combined else data["system"] + " · " + data["benchmark"]

    largest_change = float(data["change"].abs().max())
    extent = max(largest_change, 0.5)  # the colours run from -extent to +extent
    strong = extent * 0.6  # past this a cell is dark enough for white text

    # Every family is named: the labels are slanted, and none is dropped for running into its neighbour.
    x = alt.X(
        "family:N",
        title="Rule family switched off",
        sort=list(families),
        axis=alt.Axis(labelAngle=-40, orient="top", labelOverlap=False),
    )
    y = alt.Y("row:N", title=None, sort=unique(data["row"]), axis=alt.Axis(labelLimit=480))
    tooltip = [
        alt.Tooltip("system:N", title="Model"),
        alt.Tooltip("sets:Q", title="Benchmarks") if combined else alt.Tooltip("benchmark:N"),
        alt.Tooltip("family:N"),
        alt.Tooltip("change:Q", title="Change in points", format="+.2f"),
        alt.Tooltip("with every family on:Q", format=".2f"),
    ]
    base = alt.Chart(data).encode(x=x, y=y, tooltip=tooltip)

    # Red where the family helps, blue where switching it off helps, and a faint grey at no
    # change, which reads on a light page and on a dark one.
    cell_colour = alt.Color(
        "change:Q",
        title="Points of error rate",
        scale=alt.Scale(domain=[-extent, 0, extent], range=["#2f6fd0", "rgba(128,128,128,0.14)", "#d43a2f"]),
        legend=alt.Legend(orient="bottom", gradientLength=240),
    )
    cells = base.mark_rect().encode(color=cell_colour)

    # The change is written in each cell: white on a strong colour, and in the page's text colour
    # on a faint one. A change smaller than 0.05 points is not written.
    if dark:
        faint_text = "#e8e8e8"
    else:
        faint_text = "#1f1f1f"
    number = base.mark_text(fontSize=11).encode(text=alt.Text("change:Q", format="+.2f"))
    is_strong = f"abs(datum.change) > {strong}"
    is_faint = f"abs(datum.change) >= 0.05 && abs(datum.change) <= {strong}"
    on_strong = number.transform_filter(is_strong).encode(color=alt.value("white"))
    on_faint = number.transform_filter(is_faint).encode(color=alt.value(faint_text))

    return alt.layer(cells, on_strong, on_faint).properties(width=alt.Step(44), height=alt.Step(30))


# ── how a field's values are spread ──────────────────────────────────────────────────────


def _number_bars(values_by_benchmark: dict[str, list[Any]], bins: int) -> list[dict[str, Any]]:
    """A row for each bar of a number. The benchmarks share ``bins`` bars of equal width across the range of them all."""
    every_value = []
    for values in values_by_benchmark.values():
        for value in values:
            every_value.append(float(value))
    edges = np.histogram_bin_edges(np.array(every_value), bins=bins)

    rows = []
    for benchmark, values in values_by_benchmark.items():
        counts, _ = np.histogram(np.array(values, dtype=float), bins=edges)
        for start, end, count in zip(edges[:-1], edges[1:], counts, strict=True):
            if count == 0:
                continue  # an empty bar is not drawn
            rows.append(
                {
                    "benchmark": benchmark,
                    "start": float(start),
                    "end": float(end),
                    "test_cases": int(count),
                    "share": 100.0 * int(count) / len(values),
                }
            )
    return rows


def _flag_bars(values_by_benchmark: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """A row for each bar of a flag: a benchmark's test cases where it is false, and those where it is true."""
    rows = []
    for benchmark, values in values_by_benchmark.items():
        for label, count in (("false", values.count(False)), ("true", values.count(True))):
            if count == 0:
                continue
            rows.append({"benchmark": benchmark, "value": label, "test_cases": count, "share": 100.0 * count / len(values)})
    return rows


def histogram(
    values_by_benchmark: dict[str, Sequence[Any]],
    title: str,
    colours: dict[str, str],
    bins: int = 20,
    number: bool = True,
) -> alt.Chart:
    """How a field's values are spread over each benchmark's test cases.

    A number gets bars, and a flag a bar each for false and true. A bar is the share of its
    own benchmark's test cases, so a set of 600 test cases and one of 40,000 can be read on
    one chart. The test cases are counted here, and the chart carries a row per bar: a
    benchmark's every test case is never sent to the browser. For a number, the chosen
    benchmarks share ``bins`` bars of equal width across their range.

    Test cases with no value are left out: the page says how many there are. ``colours`` gives
    each benchmark its colour, so it has the same one on every chart. There is no legend on
    the chart: the page shows one for them all.
    """
    present = {}  # each benchmark's values, without the test cases that have none
    for benchmark, values in values_by_benchmark.items():
        with_a_value = [value for value in values if value is not None]
        if with_a_value:
            present[benchmark] = with_a_value
    if not present:
        raise NoData("No test case of these benchmarks has a value for this field.")

    names = list(colours)
    colour = alt.Color(
        "benchmark:N",
        sort=names,
        scale=alt.Scale(domain=names, range=[colours[name] for name in names]),
        legend=None,
    )
    share = alt.Y("share:Q", title="Share of test cases (%)", stack=None)
    tooltip = [
        alt.Tooltip("benchmark:N"),
        alt.Tooltip("test_cases:Q", title="Test cases", format=","),
        alt.Tooltip("share:Q", title="Share (%)", format=".1f"),
    ]

    if number:
        bars = pd.DataFrame(_number_bars(present, bins))
        tooltip.append(alt.Tooltip("start:Q", title="From", format=".4~g"))
        tooltip.append(alt.Tooltip("end:Q", title="To", format=".4~g"))
        chart = alt.Chart(bars).mark_bar(opacity=0.6).encode(
            x=alt.X("start:Q", bin="binned", title=title),
            x2="end:Q",
            y=share,
            color=colour,
            tooltip=tooltip,
        )
    else:
        bars = pd.DataFrame(_flag_bars(present))
        tooltip.append(alt.Tooltip("value:N", title=title))
        chart = alt.Chart(bars).mark_bar().encode(
            x=alt.X("value:N", title=title, sort=["false", "true"], axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("benchmark:N", sort=names),
            y=share,
            color=colour,
            tooltip=tooltip,
        )
    return chart.properties(height=260)
