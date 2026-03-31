"""Shared Streamlit UI components used by all benchmark pages."""

import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import torch
import streamlit as st
from omegaconf import OmegaConf

from bench.constants import BASELINE_DECODERS
from bench.params import BenchTaskParams, QECParams
from constants import (
    DEFAULT_BASELINE_DECODERS,
    DEFAULT_MAX_ITER,
    DEFAULT_P_LIST,
    DEFAULT_SHOTS_CAP,
    DEFAULT_ERRORS_CAP,
    TORCH_RUNS_ROOT,
)
from torchdecoder_utils import (
    discover_run_dirs,
    group_run_dirs_by_code_and_noise,
    group_run_dirs_by_d_rounds_basis,
    group_run_dirs_by_decoder_model_name,
    extract_pytorch_decoder_run_id,
    extract_pytorch_decoder_name,
    load_config_from_run_dir,
    flatten_config,
    get_differing_keys,
)


def render_sidebar_baselines_selection() -> list[str]:
    """Render the baseline decoders selection sidebar.

    Return the selected baseline decoders.
    """
    with st.sidebar:
        st.subheader("Select baseline decoder(s)")
        selected_baseline_decoders = st.multiselect(
            "Baseline decoder(s) to benchmark against",
            options=BASELINE_DECODERS,
            default=DEFAULT_BASELINE_DECODERS,
        )
    return selected_baseline_decoders


def render_sidebar_bench_task_selection() -> BenchTaskParams:
    """Render the benchmark task parameters selection sidebar.

    Return the selected benchmark task parameters. Call `st.stop()` if the user input
    p_list is empty or unparsable.
    """
    with st.sidebar:
        st.subheader("Select benchmark task parameters")
        max_iter = st.number_input(
            "Max number of decoding iterations",
            value=DEFAULT_MAX_ITER,
            min_value=1,
            help="Only apply to iterative decoders (e.g., BP, LearnedDMemBP).",
        )
        p_list_str = st.text_input(
            "Physical error rates (comma-separated)",
            value=", ".join(str(p) for p in DEFAULT_P_LIST),
            help="List of physical error rates to benchmark at, separated by commas.",
        )
        try:
            p_list = [float(x.strip()) for x in p_list_str.split(",") if x.strip()]
        except ValueError:
            st.error("Cannot parse into a list of floats.")
            st.stop()
        if len(p_list) == 0:
            st.warning("Please enter at least one physical error rate.")
            st.stop()
        use_prior_in_ckpt = st.checkbox(
            "Use prior from checkpoint",
            value=True,
            help="If checked, use the prior from the checkpoint; otherwise, use the prior "
            "derived from the physical error rate. Only applies to PyTorch decoders.",
        )
    return BenchTaskParams(
        max_iter=max_iter,
        p_list=p_list,
        use_prior_in_ckpt=use_prior_in_ckpt,
    )


def render_sidebar_collector_selection_common() -> tuple[int, int, str]:
    """Render the Monte Carlo collector parameters selection sidebar.

    Return `shots_cap`, `errors_cap`, `device`.
    """
    with st.sidebar:
        shots_cap = st.number_input(
            "Shots cap",
            value=DEFAULT_SHOTS_CAP,
            min_value=1,
            help="Stop Monte Carlo sampling after taking this many shots.",
        )
        errors_cap = st.number_input(
            "Logical errors cap",
            value=DEFAULT_ERRORS_CAP,
            min_value=1,
            help="Stop Monte Carlo sampling after having seen this many logical errors.",
        )
        device = st.selectbox(
            "Device for PyTorch",
            options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            index=0,
            help="Device to run PyTorch decoders on. (Baseline decoders are always run on CPU.)",
        )
    return shots_cap, errors_cap, device


def render_qec_selection() -> tuple[QECParams, list[Path]]:
    """Discover trained torch decoders and render QEC parameter selections.

    Return the selected QEC parameters and the list of run directories matching
    those parameters. Call `st.stop()` if no trained torch decoders are found
    or nothing is selected yet.
    """
    run_dirs = discover_run_dirs(TORCH_RUNS_ROOT)
    if len(run_dirs) == 0:
        st.warning("No trained PyTorch decoders found. Train a model first.")
        st.stop()

    st.subheader("Select QEC parameters")
    col1, col2 = st.columns(2)
    with col1:
        grouped = group_run_dirs_by_code_and_noise(run_dirs)
        code_noise_pairs = sorted(grouped.keys())
        selected_code_noise_pair = st.selectbox(
            "code, noise model",
            options=code_noise_pairs,
            index=None,
            format_func=lambda x: f"{x[0]}, {x[1]}",
        )
    if selected_code_noise_pair is None:
        st.stop()
    code, noise_model = selected_code_noise_pair
    run_dirs = grouped[selected_code_noise_pair]

    with col2:
        grouped = group_run_dirs_by_d_rounds_basis(run_dirs)
        d_rounds_basis_triples = sorted(grouped.keys())
        selected_d_rounds_basis_triple = st.selectbox(
            "d, rounds, basis",
            options=d_rounds_basis_triples,
            index=None,
            format_func=lambda x: f"{x[0]}, {x[1]}, {x[2]}",
        )
    if selected_d_rounds_basis_triple is None:
        st.stop()
    d, rounds, basis = selected_d_rounds_basis_triple
    run_dirs = grouped[selected_d_rounds_basis_triple]

    qec_params = QECParams(
        code=code,
        noise_model=noise_model,
        d=d,
        rounds=rounds,
        basis=basis,
    )
    return qec_params, run_dirs


def render_decoder_selection(run_dirs: list[Path]) -> list[Path]:
    """Render torch decoder tables with row selection and config expanders.

    Return the list of selected run directories.
    """
    st.subheader("Select PyTorch decoder(s)")
    st.caption(
        "One table per decoder model. "
        "Only differing config fields are shown in each table. "
        "Click the checkboxes to select runs to benchmark. "
        "Click the expander below to view full configs."
    )
    grouped = group_run_dirs_by_decoder_model_name(run_dirs)
    selected_run_dirs: list[Path] = []
    for model_name in sorted(grouped.keys()):
        group = sorted(grouped[model_name], key=extract_pytorch_decoder_run_id)
        df_data: dict[str, list[str]] = {"Run ID": [r.name for r in group]}
        if len(group) >= 2:
            configs = [load_config_from_run_dir(r) for r in group]
            flat_configs = [
                flatten_config(
                    cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
                )
                for cfg in configs
            ]
            diff_keys = get_differing_keys(flat_configs)
            df_data.update(
                {
                    k: [str(c.get(k, "N/A")) for c in flat_configs]
                    for k in sorted(diff_keys)
                }
            )
        df = pd.DataFrame(df_data)

        st.markdown(f"**{model_name}**")
        event = st.dataframe(
            df,
            width="stretch",
            key=f"pytorch_decoder_selection_{model_name}",
            on_select="rerun",
            selection_mode="multi-row",
            hide_index=True,
        )
        selected_indices = event.selection.rows or []  # type: ignore
        selected_run_dirs.extend([group[i] for i in selected_indices])

        with st.expander("View full configs"):
            selected_view_config = st.selectbox(
                f"{model_name} config",
                options=group,
                index=None,
                format_func=lambda r: r.name,
                key=f"config_viewer_{model_name}",
                placeholder="Choose a run ID",
            )
            if selected_view_config:
                cfg = load_config_from_run_dir(selected_view_config)
                cfg_yaml = OmegaConf.to_yaml(cfg)
                st.code(cfg_yaml, language="yaml")

    return selected_run_dirs


def stop_if_no_decoders_selected(
    selected_run_dirs: list[Path],
    selected_baseline_decoders: list[str],
) -> None:
    """Call `st.stop()` if no decoders are selected."""
    if len(selected_run_dirs) == 0 and len(selected_baseline_decoders) == 0:
        st.warning(
            "Please select at least one PyTorch decoder or baseline decoder to benchmark."
        )
        st.stop()


def render_missing_data_warning_and_benchmark_button(
    pending_run_dirs: list[Path],
    pending_baseline_decoders: list[str],
) -> bool:
    """Show a warning for missing data and a 'Run benchmark' button.

    Return True if the button was clicked.
    """
    pending_list = [
        extract_pytorch_decoder_name(r) for r in pending_run_dirs
    ] + pending_baseline_decoders
    pending_list_str = "\n\n".join(f"• {d}" for d in pending_list)
    st.warning(
        "The following decoders have missing or incomplete benchmark data:\n\n"
        f"{pending_list_str}\n\n"
        "Please run benchmark first."
    )
    clicked = st.button("Run benchmark", type="primary")
    return clicked


# ---------------------------------------------------------------------------
# Benchmark modal dialog
# ---------------------------------------------------------------------------

_BENCH_THREAD_KEY = "_bench_thread"
_BENCH_STOP_EVENT_KEY = "_bench_stop_event"
_BENCH_STATE_KEY = "_bench_state"


def _cleanup_bench_session_state() -> None:
    """Remove all benchmark-related keys from session state."""
    for key in (_BENCH_THREAD_KEY, _BENCH_STOP_EVENT_KEY, _BENCH_STATE_KEY):
        st.session_state.pop(key, None)


def is_benchmark_running() -> bool:
    """Return True if a benchmark thread is tracked in session state."""
    return _BENCH_THREAD_KEY in st.session_state


def start_benchmark_thread(
    target_fn: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Start `target_fn` in a daemon thread and store handles in session state.

    `target_fn` is called as `target_fn(stop_event, *args, **kwargs)` where
    `stop_event` is a `threading.Event` that the function should check
    periodically so it can exit early when "Stop benchmark" button is clicked.
    """
    stop_event = threading.Event()
    bench_state: dict[str, str | None] = {"error": None}

    def _worker() -> None:
        try:
            target_fn(stop_event, *args, **kwargs)
        except Exception:
            bench_state["error"] = traceback.format_exc()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    st.session_state[_BENCH_THREAD_KEY] = thread
    st.session_state[_BENCH_STOP_EVENT_KEY] = stop_event
    st.session_state[_BENCH_STATE_KEY] = bench_state


@st.dialog("Running Benchmark", dismissible=False)
def benchmark_modal() -> None:
    """Modal dialog shown while a benchmark is in progress.

    Blocks all interaction with the rest of the app.  The only way to dismiss
    the dialog is by clicking the "Stop benchmark" button or waiting for the
    benchmark to finish.
    """
    thread: threading.Thread = st.session_state[_BENCH_THREAD_KEY]
    stop_event: threading.Event = st.session_state[_BENCH_STOP_EVENT_KEY]
    bench_state: dict[str, str | None] = st.session_state[_BENCH_STATE_KEY]

    if thread.is_alive():
        st.info("Benchmark is running. Please wait...")
        if st.button("Stop benchmark", type="primary", use_container_width=True):
            stop_event.set()
            _cleanup_bench_session_state()
            st.rerun(scope="app")
        time.sleep(2)
        st.rerun()
    else:
        thread.join()
        error = bench_state.get("error")
        if error:
            st.error(f"Benchmark failed:\n\n```\n{error}\n```")
            if st.button("Close", use_container_width=True):
                _cleanup_bench_session_state()
                st.rerun(scope="app")
        else:
            _cleanup_bench_session_state()
            st.rerun(scope="app")
