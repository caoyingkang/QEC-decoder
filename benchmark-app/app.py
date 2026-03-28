"""
Streamlit app for Monte Carlo benchmarking of QEC decoders.
Run with: `uv run streamlit run benchmark-app/app.py` (from repo root)
"""

import streamlit as st

pg = st.navigation(
    {
        "Benchmark Tool": [
            st.Page("pages/custom_benchmark_page.py", title="Customized", default=True),
            st.Page("pages/sinter_benchmark_page.py", title="Sinter"),
        ]
    },
    position="sidebar",
)
st.set_page_config(page_title="Decoder Benchmark", layout="wide", page_icon="📈")
st.title("Monte Carlo Benchmark")
pg.run()
