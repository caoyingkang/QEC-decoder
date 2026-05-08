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
from bench.params import BenchTaskParams
from constants import BASELINES_CSV_DIR, TORCH_RUNS_ROOT
from plotting import render_plot
from shared_ui import (
    render_baselines_selection,
    render_p_list_selection,
    render_sidebar_collector_selection_common,
    render_qec_selection,
    validate_stim_files,
    render_torchdecoder_selection,
    stop_if_no_decoders_selected,
    render_missing_data_warning_and_benchmark_button,
)
from torchdecoder_utils import (
    discover_run_dirs,
    extract_pytorch_decoder_name,
    load_model_config_from_run_dir,
    get_ckpt_path,
)
from experiment_factory import create_experiment


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


# -- Page layout ---------------------------------------------------------------

p_list = render_p_list_selection()
collector_params = _render_sidebar_collector_selection()

qec_params = render_qec_selection()
load_circuit_from_file = (
    qec_params.code.startswith("BB_") or qec_params.code == "HexColorCode"
)
if load_circuit_from_file:
    validate_stim_files(qec_params, p_list)

selected_baseline_decoders, baseline_decoder_params = render_baselines_selection(
    qec_params
)

run_dirs = discover_run_dirs(TORCH_RUNS_ROOT, qec_params)

if len(run_dirs) > 0:
    selected_run_dirs, torchdecoder_shared_params = render_torchdecoder_selection(
        run_dirs, qec_params
    )
else:
    selected_run_dirs = []
    torchdecoder_shared_params = {}

stop_if_no_decoders_selected(selected_run_dirs, selected_baseline_decoders)

benchtask_params = BenchTaskParams(
    p_list=p_list,
    baseline_decoder_params=baseline_decoder_params,
    torchdecoder_shared_params=torchdecoder_shared_params,
)

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
            # Build experiments: either by loading from stim file paths or by creating from QEC parameters
            experiments = {
                p: create_experiment(
                    qec_params,
                    p,
                    load_circuit_from_file=load_circuit_from_file,
                )
                for p in benchtask_params.p_list
            }

            for baseline_decoder in pending_baseline_decoders:
                run_baseline_benchmark(
                    BASELINES_CSV_DIR,
                    baseline_decoder,
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
                    experiments=experiments,
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
                    experiments=experiments,
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
