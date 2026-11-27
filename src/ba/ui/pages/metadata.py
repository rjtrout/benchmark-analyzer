"""Benchmark metadata: how each metadata field is spread over the test cases of the chosen benchmarks.

One histogram per field. Fields are either given by the source or derived here, and the
page can show either kind or both.
"""

from __future__ import annotations

import streamlit as st

from ba.metadata.derived import FLAGS, FROM_AUDIO
from ba.ui import charts, controls, page, queries


def field_panel(
    project: page.Project,
    field_id: str,
    source: str,
    what_it_is: str,
    benchmark_ids: list[str],
    colours: dict[str, str],
    bins: int,
) -> None:
    """One field: what it is, and how its values are spread over each chosen benchmark's test cases."""
    st.markdown(f"**{field_id}** · {source}")
    st.caption(what_it_is)

    have = {}  # each benchmark's values
    for benchmark_id in benchmark_ids:
        have[benchmark_id] = queries.field_values(project.root, project.stamp, field_id, benchmark_id)

    try:
        chart = charts.histogram(have, field_id, colours, bins, number=field_id not in FLAGS)
    except charts.NoData:
        note = "No test case of these benchmarks has a value for it."
        if field_id in FROM_AUDIO:
            note += " It is measured on the audio, which `ba fetch --audio` fetches."
        st.caption(note)
        return
    st.altair_chart(chart)

    in_all = 0
    with_a_value = 0
    for values in have.values():
        in_all += len(values)
        with_a_value += len([value for value in values if value is not None])
    st.caption(f"{with_a_value:,} of {in_all:,} test cases have a value.")


def main() -> None:
    project = page.frame()
    st.title("Benchmark metadata")
    st.caption(
        "How each metadata field is spread over the test cases. A bar is the share of its own benchmark's test cases, so sets of any size can be "
        "read together. A field is either given by the source or derived here from the test case."
    )

    listed = controls.benchmarks_with_lists(project)
    if not listed:
        st.info("No benchmark has a test case list yet. `ba fetch` reads them.")
        return

    first, second, third = st.columns([5, 2, 1])
    benchmark_ids = first.multiselect("Benchmarks", listed, default=listed, key="metadata_benchmarks")
    sources = second.multiselect(
        "Fields",
        list(controls.SOURCES),
        default=list(controls.SOURCES),
        key="metadata_sources",
        help="Given: stated by the dataset. Derived: worked out here, from the text or the audio.",
    )
    bins = third.number_input(
        "Bins",
        min_value=5,
        max_value=60,
        value=20,
        step=5,
        key="metadata_bins",
        help="For a number: how many bars of equal width, across the range of the chosen benchmarks.",
    )
    if not benchmark_ids or not sources:
        st.info("Pick at least one benchmark and one kind of field.")
        return

    # One legend for every chart on the page: each benchmark's colour, and how many test cases it has.
    colours = charts.benchmark_colours(benchmark_ids)
    legend = {}
    for benchmark_id in benchmark_ids:
        how_many = len(project.store.test_cases.ids(benchmark_id))
        legend[f"{benchmark_id} {how_many:,} test cases"] = colours[benchmark_id]
    page.legend(legend)

    # The fields of the chosen kinds, in two columns.
    shown = []  # each field: its id, where it comes from, and what it is
    for source, fields in controls.SOURCES.items():
        if source in sources:
            for field_id, what_it_is in fields.items():
                shown.append((field_id, source.lower(), what_it_is))
    left, right = st.columns(2)
    for index, (field_id, source, what_it_is) in enumerate(shown):
        if index % 2 == 0:
            column = left
        else:
            column = right
        with column.container(border=True):
            field_panel(project, field_id, source, what_it_is, benchmark_ids, colours, int(bins))


with page.guard():
    main()
