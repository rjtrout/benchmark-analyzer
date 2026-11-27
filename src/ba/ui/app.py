"""The web UI's entry point.

Run it with ``ba ui`` or ``streamlit run src/ba/ui/app.py``. It reads the project that
``Paths.discover()`` finds: ``$BA_ROOT``, else the repo this file is in.

Four pages: Results, Normalization, Slices and Benchmark metadata. Each page is a script
that draws the shared sidebar itself, through ``ba.ui.page.frame``.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Benchmark analyzer", layout="wide")
st.navigation(
    [
        st.Page("pages/results.py", title="Results", default=True),
        st.Page("pages/normalization.py", title="Normalization"),
        st.Page("pages/slices.py", title="Slices"),
        st.Page("pages/metadata.py", title="Benchmark metadata"),
    ]
).run()
