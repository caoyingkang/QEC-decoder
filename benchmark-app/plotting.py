"""Shared Streamlit plotting helpers."""

from io import BytesIO

import matplotlib.figure
import plotly.graph_objects as go
import streamlit as st


def render_plot(fig: matplotlib.figure.Figure, filename: str) -> None:
    """Display a matplotlib figure in Streamlit and offer a PNG download button."""
    _, plot_col, _ = st.columns([1, 3, 1])
    with plot_col:
        st.pyplot(fig, width="stretch")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    st.download_button(
        "Download plot as PNG",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
    )


def render_plotly(fig: go.Figure) -> None:
    """Display a Plotly figure in Streamlit."""
    st.plotly_chart(fig)
