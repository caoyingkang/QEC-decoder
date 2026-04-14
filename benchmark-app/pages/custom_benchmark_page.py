"""Custom Monte Carlo benchmark page."""

import os
import threading
from pathlib import Path
import queue

import streamlit as st

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
from bench.custom_bench.baselines_runner import run_baseline_benchmark
from bench.custom_bench.torchdecoder_runner import run_torchdecoder_benchmark
from bench.params import BenchTaskParams, QECParams
from constants import DEFAULT_BATCH_SIZE, BASELINES_CSV_DIR, TORCH_RUNS_ROOT
from plotting import render_plot, render_plotly
from qecdec.experiments import Experiment
from shared_ui import (
    render_baselines_selection,
    render_circuit_source_selection,
    render_p_list_selection,
    render_sidebar_collector_selection_common,
    render_qec_selection,
    render_stim_file_selection,
    render_torchdecoder_selection,
    stop_if_no_decoders_selected,
    render_missing_data_warning_and_benchmark_button,
)
from torchdecoder_utils import (
    discover_run_dirs,
    extract_pytorch_decoder_name,
    group_run_dirs_by_code_and_noise,
    group_run_dirs_by_d_rounds_basis,
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


def _run_all_benchmarks(
    stop_event: threading.Event,
    exc_queue: queue.Queue,
    pending_baseline_decoders: list[str],
    pending_run_dirs: list[Path],
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    experiments: dict[float, Experiment] | None = None,
) -> None:
    """Run all pending benchmark tasks. Exit early if `stop_event` is set.
    Put any exception that occurs into `exc_queue`."""
    try:
        for baseline_decoder in pending_baseline_decoders:
            if stop_event.is_set():
                return
            run_baseline_benchmark(
                BASELINES_CSV_DIR,
                baseline_decoder,
                qec_params=qec_params,
                benchtask_params=benchtask_params,
                collector_params=collector_params,
                stop_event=stop_event,
                experiments=experiments,
            )
        for run_dir in pending_run_dirs:
            if stop_event.is_set():
                return
            run_torchdecoder_benchmark(
                csv_path=get_torchdecoder_csv_path(run_dir),
                decoder_name=extract_pytorch_decoder_name(run_dir),
                model_cfg=load_model_config_from_run_dir(run_dir),
                ckpt_path=get_ckpt_path(run_dir),
                qec_params=qec_params,
                benchtask_params=benchtask_params,
                collector_params=collector_params,
                stop_event=stop_event,
                experiments=experiments,
            )
    except Exception as e:
        exc_queue.put(e)


@st.fragment(run_every=2)
def _stop_monitor(
    thread: threading.Thread,
    stop_event: threading.Event,
    exc_queue: queue.Queue,
) -> None:
    """Monitor the benchmark thread and stop event."""
    st.info("Benchmark is running. Please wait...")
    if not thread.is_alive():
        if not exc_queue.empty():
            raise exc_queue.get()
        st.rerun(scope="app")
    if st.button("Stop benchmark", type="primary", use_container_width=True):
        stop_event.set()
        with st.spinner("Stopping benchmark..."):
            thread.join()
        st.rerun(scope="app")


@st.dialog("Running Benchmark", dismissible=False)
def _benchmark_progress_modal(
    pending_baseline_decoders: list[str],
    pending_run_dirs: list[Path],
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    load_circuit_from_file: bool,
) -> None:
    """Modal dialog shown while a benchmark is in progress.

    Blocks all interaction with the rest of the app.  The only way to dismiss
    the dialog is by clicking the "Stop benchmark" button or waiting for the
    benchmark to finish.
    """
    # Build experiments: either by loading from stim file paths or by creating from QEC parameters
    experiments = {
        p: create_experiment(
            qec_params,
            p,
            load_circuit_from_file=load_circuit_from_file,
        )
        for p in benchtask_params.p_list
    }

    stop_event = threading.Event()
    exc_queue = queue.Queue()
    thread = threading.Thread(
        target=_run_all_benchmarks,
        args=(
            stop_event,
            exc_queue,
            pending_baseline_decoders,
            pending_run_dirs,
            qec_params,
            benchtask_params,
            collector_params,
            experiments,
        ),
        daemon=True,
    )
    thread.start()
    _stop_monitor(thread, stop_event, exc_queue)


# -- Page layout ---------------------------------------------------------------

p_list = render_p_list_selection()
collector_params = _render_sidebar_collector_selection()

circuit_source = render_circuit_source_selection()
load_circuit_from_file = circuit_source == "stim_file"

if load_circuit_from_file:
    qec_params, _ = render_stim_file_selection(p_list)
    # Discover matching torch run dirs if any exist
    all_run_dirs = discover_run_dirs(TORCH_RUNS_ROOT)
    run_dirs: list[Path] = []
    if len(all_run_dirs) > 0:
        grouped = group_run_dirs_by_code_and_noise(all_run_dirs)
        key = (qec_params.code, qec_params.noise_model)
        if key in grouped:
            grouped = group_run_dirs_by_d_rounds_basis(grouped[key])
            key = (qec_params.d, qec_params.rounds, qec_params.basis)
            if key in grouped:
                run_dirs = grouped[key]
else:
    qec_params, run_dirs = render_qec_selection()

selected_baseline_decoders, baseline_decoder_params = render_baselines_selection(
    qec_params
)

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
        _benchmark_progress_modal(
            pending_baseline_decoders,
            pending_run_dirs,
            qec_params=qec_params,
            benchtask_params=benchtask_params,
            collector_params=collector_params,
            load_circuit_from_file=load_circuit_from_file,
        )
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
