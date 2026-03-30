"""Sinter-based Monte Carlo benchmark page."""

import os

import streamlit as st

from bench.sinter_bench.collector_params import CollectorParams
from bench.sinter_bench.plotting import plot_ler_vs_per
from bench.sinter_bench.stats_io import (
    get_torchdecoder_csv_path,
    load_and_merge_stats,
)
from bench.sinter_bench.baselines_runner import run_baseline_benchmark
from bench.sinter_bench.torchdecoder_runner import run_torchdecoder_benchmark
from torchdecoder_utils import (
    extract_pytorch_decoder_name,
    load_model_config_from_run_dir,
    get_ckpt_path,
)
from constants import BASELINES_CSV_DIR
from plotting import render_plot
from shared_ui import (
    render_sidebar_baselines_selection,
    render_sidebar_bench_task_selection,
    render_sidebar_collector_selection_common,
    render_qec_selection,
    render_decoder_selection,
    stop_if_no_decoders_selected,
    render_missing_data_warning_and_benchmark_button,
)


def _render_sidebar_collector_selection() -> CollectorParams:
    """Render the Monte Carlo collector parameters selection sidebar.

    Return the selected Monte Carlo collector parameters.
    """
    with st.sidebar:
        st.subheader("Select Monte Carlo collector parameters")

    shots_cap, errors_cap, device = render_sidebar_collector_selection_common()

    with st.sidebar:
        num_workers = st.number_input(
            "Number of workers",
            value=max(1, (os.cpu_count() or 1) - 1),
            min_value=1,
            help="Number of workers to launch sinter.collect().",
        )

    return CollectorParams(
        shots_cap=shots_cap,
        errors_cap=errors_cap,
        device=device,
        num_workers=num_workers,
    )


selected_baseline_decoders = render_sidebar_baselines_selection()
benchtask_params = render_sidebar_bench_task_selection()
collector_params = _render_sidebar_collector_selection()
qec_params, run_dirs = render_qec_selection()
selected_run_dirs = render_decoder_selection(run_dirs, qec_params)
stop_if_no_decoders_selected(selected_run_dirs, selected_baseline_decoders)

# Load and merge statistics
stats, pending_run_dirs, pending_baseline_decoders = load_and_merge_stats(
    selected_run_dirs,
    selected_baseline_decoders,
    BASELINES_CSV_DIR,
    benchtask_params=benchtask_params,
    collector_params=collector_params,
    qec_params=qec_params,
)

# Handle missing data
if len(pending_run_dirs) > 0 or len(pending_baseline_decoders) > 0:
    clicked = render_missing_data_warning_and_benchmark_button(
        pending_run_dirs, pending_baseline_decoders
    )
    if clicked:
        with st.spinner("Running benchmark..."):
            for baseline_decoder in pending_baseline_decoders:
                run_baseline_benchmark(
                    BASELINES_CSV_DIR,
                    baseline_decoder,
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
                )
            for run_dir in pending_run_dirs:
                run_torchdecoder_benchmark(
                    csv_path=get_torchdecoder_csv_path(run_dir),
                    decoder_name=extract_pytorch_decoder_name(run_dir),
                    model_cfg=load_model_config_from_run_dir(run_dir),
                    ckpt_path=get_ckpt_path(run_dir),
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
                )
            st.rerun()
    st.stop()


# Data is complete -- plot
st.subheader("Logical Error Rate (LER) vs Physical Error Rate (PER)")
ler_mode = st.radio(
    "LER calculation method",
    options=["per shot", "per round"],
    horizontal=True,
)

fig = plot_ler_vs_per(
    stats,
    qec_params=qec_params,
    ler_mode=ler_mode,
)
render_plot(fig, filename="benchmark_LER_vs_PER.png")
