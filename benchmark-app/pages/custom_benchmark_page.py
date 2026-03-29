"""Custom Monte Carlo benchmark page."""

import os

import streamlit as st

from constants import DEFAULT_BATCH_SIZE, BASELINES_CSV_DIR
from plotting import render_plot, render_plotly
from shared_ui import (
    render_sidebar_baselines_selection,
    render_sidebar_bench_task_selection,
    render_sidebar_collector_selection_common,
    render_qec_selection,
    render_decoder_selection,
    stop_if_no_decoders_selected,
    render_missing_data_warning_and_benchmark_button,
)
from torchdecoder_utils import (
    extract_pytorch_decoder_name,
    load_model_config_from_run_dir,
    get_ckpt_path,
)
from bench.custom_bench.collector_params import CollectorParams
from bench.custom_bench.plotting import (
    plot_ler_vs_per,
    plot_fr_vs_per,
    plot_smr_vs_per,
    plot_avg_iters_vs_per,
    plot_iters_distribution,
)
from bench.custom_bench.stats_io import (
    get_torchdecoder_csv_path,
    load_and_merge_stats,
)
from bench.custom_bench.baselines_runner import (
    run_MWPM_benchmark,
    run_BP_benchmark,
)
from bench.custom_bench.torchdecoder_runner import run_torchdecoder_benchmark


def _render_sidebar_collector_selection() -> CollectorParams:
    """Render the Monte Carlo collector parameters selection sidebar.

    Return the selected Monte Carlo collector parameters.
    """
    with st.sidebar:
        st.subheader("Select Monte Carlo collector parameters")
        batch_size = st.number_input(
            "Batch size",
            value=DEFAULT_BATCH_SIZE,
            min_value=1,
            help="Number of shots to process in each batch.",
        )

    shots_cap, errors_cap, device = render_sidebar_collector_selection_common()

    with st.sidebar:
        multiprocessing_mode = st.checkbox(
            "Use multiprocessing",
            value=(device == "cpu"),
            help="If checked, run the Monte Carlo collection with multiprocessing.",
        )
        if multiprocessing_mode:
            num_parallel_workers = st.number_input(
                "Number of parallel workers",
                value=max(1, (os.cpu_count() or 1) - 1),
                min_value=1,
                help="Number of parallel worker processes.",
            )
            if device == "cuda":
                st.warning(
                    "It is not recommended to run Python's multiprocessing on CUDA."
                )
        else:
            num_parallel_workers = 0

    return CollectorParams(
        batch_size=batch_size,
        shots_cap=shots_cap,
        errors_cap=errors_cap,
        device=device,
        num_parallel_workers=num_parallel_workers,
    )


selected_baseline_decoders = render_sidebar_baselines_selection()
benchtask_params = render_sidebar_bench_task_selection()
collector_params = _render_sidebar_collector_selection()
qec_params, run_dirs = render_qec_selection()
selected_run_dirs = render_decoder_selection(run_dirs)
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
            if "MWPM" in pending_baseline_decoders:
                run_MWPM_benchmark(
                    BASELINES_CSV_DIR,
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
                )
            if "BP" in pending_baseline_decoders:
                run_BP_benchmark(
                    BASELINES_CSV_DIR,
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
                )
            st.rerun()
    st.stop()

# Data is complete -- plot
st.subheader("Benchmark Results")
tabs = st.tabs(["FR", "LER", "SMR", "Iters (Avg)", "Iters (Dist)"])

iterative_stats = [s for s in stats if s.metadata.is_iterative]

with tabs[0]:
    st.markdown("##### Failure Rate (FR) vs Physical Error Rate (PER)")
    fr_mode = st.radio(
        "FR calculation method",
        options=["per shot", "per round"],
        horizontal=True,
        key="fr_mode",
    )
    fig = plot_fr_vs_per(stats, qec_params=qec_params, mode=fr_mode)
    render_plot(fig, filename="benchmark_FR_vs_PER.png")

with tabs[1]:
    st.markdown("##### Logical Error Rate (LER) vs Physical Error Rate (PER)")
    ler_mode = st.radio(
        "LER calculation method",
        options=["per shot", "per round"],
        horizontal=True,
        key="ler_mode",
    )
    fig = plot_ler_vs_per(stats, qec_params=qec_params, mode=ler_mode)
    render_plot(fig, filename="benchmark_LER_vs_PER.png")

with tabs[2]:
    st.markdown("##### Syndrome Mismatch Rate (SMR) vs Physical Error Rate (PER)")

    # Filter out stats with no syndrome mismatches.
    filtered_stats = [s for s in stats if s.synd_mismatches > 0]

    if len(filtered_stats) == 0:
        st.warning("No syndrome mismatches found for the selected decoders.")
    else:
        smr_mode = st.radio(
            "SMR calculation method",
            options=["per shot", "per round"],
            horizontal=True,
            key="smr_mode",
        )
        fig = plot_smr_vs_per(filtered_stats, qec_params=qec_params, mode=smr_mode)
        render_plot(fig, filename="benchmark_SMR_vs_PER.png")

with tabs[3]:
    st.markdown("##### Average Iterations vs Physical Error Rate (PER)")
    if len(iterative_stats) == 0:
        st.warning("No iterative decoders selected.")
    else:
        avg_over = st.radio(
            "Average over",
            options=["all shots", "converged shots", "successful shots"],
            horizontal=True,
            key="avg_over",
        )
        fig = plot_avg_iters_vs_per(
            iterative_stats, qec_params=qec_params, avg_over=avg_over
        )
        render_plot(fig, filename="benchmark_AvgIters_vs_PER.png")

with tabs[4]:
    st.markdown("##### Iterations Distributions")
    if len(iterative_stats) == 0:
        st.warning("No iterative decoders selected.")
    else:
        dist_over = st.radio(
            "Distribution over",
            options=["all shots", "converged shots", "successful shots"],
            horizontal=True,
            key="iter_dist_over",
        )
        for p in benchtask_params.p_list:
            fig = plot_iters_distribution(
                iterative_stats, p=p, qec_params=qec_params, dist_over=dist_over
            )
            render_plotly(fig)
