"""
Streamlit app for Monte Carlo benchmarking of QEC decoders.
Run with: `uv run streamlit run benchmark-app/app.py` (from repo root)
"""

import logging

import streamlit as st

# Suppress noisy warnings from multiprocessing workers that don't run under `streamlit run`.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)


class _SuppressBareRunWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "to view a Streamlit app on a browser" not in record.getMessage()


logging.getLogger("streamlit").addFilter(_SuppressBareRunWarning())

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
