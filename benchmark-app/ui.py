"""Streamlit UI components for the benchmark app."""

import os
from pathlib import Path
from typing import Any

from qecbench import CollectorParams, TaskMetadata, TaskStats
import streamlit as st

from constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TASK_PARAMS,
    DEFAULT_ERRORS_CAP,
    DEFAULT_SHOTS_CAP,
    REPO_ROOT,
)
from utils import dict_to_str, discover_torch_run_dirs


def render_sidebar_collector_selection() -> CollectorParams:
    """Render Monte Carlo collector parameters selection in the sidebar."""
    with st.sidebar:
        st.subheader("Select Monte Carlo collector parameters")
        batch_size = st.number_input(
            "Batch size",
            key="batch_size",
            value=DEFAULT_BATCH_SIZE,
            min_value=1,
            help="Number of shots to process in each batch.",
        )
        shots_cap = st.number_input(
            "Shots cap",
            key="shots_cap",
            value=DEFAULT_SHOTS_CAP,
            min_value=1,
            help="Stop Monte Carlo sampling after taking this many shots.",
        )
        errors_cap = st.number_input(
            "Logical errors cap",
            key="errors_cap",
            value=DEFAULT_ERRORS_CAP,
            min_value=1,
            help="Stop Monte Carlo sampling after having seen this many logical errors.",
        )
        num_parallel_workers = st.number_input(
            "Number of parallel workers",
            key="num_parallel_workers",
            value=(os.cpu_count() or 1) - 1,
            min_value=0,
            help="Number of parallel worker processes. Ignored when running on GPU devices.",
        )

    return CollectorParams(
        batch_size=batch_size,
        shots_cap=shots_cap,
        errors_cap=errors_cap,
        num_parallel_workers=num_parallel_workers,
    )


def _render_bp_params_selection(default: dict[str, Any]) -> dict[str, Any]:
    with st.expander("BP decoder params", expanded=True):
        col1, _, _ = st.columns(3)
        with col1:
            max_iter = st.number_input(
                "max_iter",
                key="bp_max_iter",
                value=default["max_iter"],
                min_value=1,
                help="Max number of BP iterations.",
            )
    return {"max_iter": max_iter}


def _render_bposd_params_selection(default: dict[str, Any]) -> dict[str, Any]:
    with st.expander("BPOSD decoder params", expanded=True):
        col1, col2, _ = st.columns(3)
        with col1:
            max_bp_iter = st.number_input(
                "max_bp_iter",
                key="bposd_max_bp_iter",
                value=default["max_bp_iter"],
                min_value=1,
                help="Max number of BP iterations.",
            )
        with col2:
            osd_order = st.number_input(
                "osd_order",
                key="bposd_osd_order",
                value=default["osd_order"],
                min_value=0,
                help="OSD order.",
            )
        return {
            "max_bp_iter": max_bp_iter,
            "osd_order": osd_order,
        }


def _render_membp_params_selection(default: dict[str, Any]) -> dict[str, Any]:
    with st.expander("MemBP decoder params", expanded=True):
        col1, col2, _ = st.columns(3)
        with col1:
            max_iter = st.number_input(
                "max_iter",
                key="membp_max_iter",
                value=default["max_iter"],
                min_value=1,
                help="Max number of BP iterations.",
            )
        with col2:
            gamma = st.number_input(
                "gamma",
                key="membp_gamma",
                value=default["gamma"],
                help="Memory coefficient.",
            )
    return {"max_iter": max_iter, "gamma": gamma}


def _render_relaybp_params_selection(default: dict[str, Any]) -> dict[str, Any]:
    with st.expander("RelayBP decoder params", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            gamma0 = st.number_input(
                "gamma0",
                key="relaybp_gamma0",
                value=default["gamma0"],
                format="%f",
                help="Memory parameter for the initial MemBP stage.",
            )
        with col2:
            gdi_low = st.number_input(
                "gamma_dist_interval low",
                key="relaybp_gdi_low",
                value=default["gamma_dist_interval"][0],
                format="%f",
                help="Lower bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        with col3:
            gdi_high = st.number_input(
                "gamma_dist_interval high",
                key="relaybp_gdi_high",
                value=default["gamma_dist_interval"][1],
                format="%f",
                help="Upper bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        if gdi_low >= gdi_high:
            st.error("gamma_dist_interval: low must be strictly less than high.")
            st.stop()
        col4, col5, col6 = st.columns(3)
        with col4:
            num_relays = st.number_input(
                "num_relays",
                key="relaybp_num_relays",
                value=default["num_relays"],
                min_value=1,
                help="Number of DMemBP relays (beyond the initial stage).",
            )
        with col5:
            pre_iter = st.number_input(
                "pre_iter",
                key="relaybp_pre_iter",
                value=default["pre_iter"],
                min_value=1,
                help="Number of iterations for the initial stage.",
            )
        with col6:
            max_iter_per_relay = st.number_input(
                "max_iter_per_relay",
                key="relaybp_max_iter_per_relay",
                value=default["max_iter_per_relay"],
                min_value=1,
                help="Max number of iterations per DMemBP relay.",
            )
        col7, _, _ = st.columns(3)
        with col7:
            stop_nconv = st.number_input(
                "stop_nconv",
                key="relaybp_stop_nconv",
                value=default["stop_nconv"],
                min_value=1,
                max_value=num_relays + 1,
                help="How many solutions to find before terminating.",
            )
    return {
        "gamma0": gamma0,
        "gamma_dist_interval": (gdi_low, gdi_high),
        "num_relays": num_relays,
        "pre_iter": pre_iter,
        "max_iter_per_relay": max_iter_per_relay,
        "stop_nconv": stop_nconv,
    }


def _render_multirelaybp_params_selection(default: dict[str, Any]) -> dict[str, Any]:
    with st.expander("MultiRelayBP decoder params", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            gamma0 = st.number_input(
                "gamma0",
                key="multirelaybp_gamma0",
                value=default["gamma0"],
                format="%f",
                help="Memory parameter for the initial MemBP stage.",
            )
        with col2:
            gdi_low = st.number_input(
                "gamma_dist_interval low",
                key="multirelaybp_gdi_low",
                value=default["gamma_dist_interval"][0],
                format="%f",
                help="Lower bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        with col3:
            gdi_high = st.number_input(
                "gamma_dist_interval high",
                key="multirelaybp_gdi_high",
                value=default["gamma_dist_interval"][1],
                format="%f",
                help="Upper bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        if gdi_low >= gdi_high:
            st.error("gamma_dist_interval: low must be strictly less than high.")
            st.stop()
        col4, col5, col6 = st.columns(3)
        with col4:
            num_chains = st.number_input(
                "num_chains",
                key="multirelaybp_num_chains",
                value=default["num_chains"],
                min_value=1,
                help="Number of independent chains after the initial stage.",
            )
        with col5:
            num_relays = st.number_input(
                "num_relays",
                key="multirelaybp_num_relays",
                value=default["num_relays"],
                min_value=1,
                help="Number of DMemBP relays in each chain (beyond the initial stage).",
            )
        with col6:
            pre_iter = st.number_input(
                "pre_iter",
                key="multirelaybp_pre_iter",
                value=default["pre_iter"],
                min_value=1,
                help="Number of iterations for the initial stage.",
            )
        col7, col8, _ = st.columns(3)
        with col7:
            max_iter_per_relay = st.number_input(
                "max_iter_per_relay",
                key="multirelaybp_max_iter_per_relay",
                value=default["max_iter_per_relay"],
                min_value=1,
                help="Max number of iterations per DMemBP relay.",
            )
        with col8:
            stop_nconv = st.number_input(
                "stop_nconv",
                key="multirelaybp_stop_nconv",
                value=default["stop_nconv"],
                min_value=1,
                max_value=num_relays + 1,
                help="How many solutions to find before each chain terminates.",
            )
    return {
        "gamma0": gamma0,
        "gamma_dist_interval": (gdi_low, gdi_high),
        "num_chains": num_chains,
        "num_relays": num_relays,
        "pre_iter": pre_iter,
        "max_iter_per_relay": max_iter_per_relay,
        "stop_nconv": stop_nconv,
    }


def _render_learned_dmembp_params_selection(
    default: dict[str, Any], circuit_name: str, circuit_params: dict[str, Any]
) -> dict[str, Any]:
    all_run_dirs = discover_torch_run_dirs(
        circuit_name, circuit_params, "LearnedDMemBP"
    )
    with st.expander("LearnedDMemBP decoder params", expanded=True):
        col1, col2, _ = st.columns(3)
        with col1:
            max_iter = st.number_input(
                "max_iter",
                key="learned_dmembp_max_iter",
                value=default["max_iter"],
                min_value=1,
                help="Max number of BP iterations.",
            )
        with col2:
            run_dir = st.selectbox(
                "checkpoint",
                key="learned_dmembp_checkpoint",
                options=all_run_dirs,
                index=None,
                format_func=lambda x: x.name,
                help="Which checkpoint to load gamma vector from.",
            )
    if run_dir is None:
        st.stop()
    ckpt_path = run_dir / "checkpoints" / "best_model.ckpt"
    ckpt_rel_path = ckpt_path.relative_to(REPO_ROOT).as_posix()

    return {"max_iter": max_iter, "ckpt_rel_path": ckpt_rel_path}


def _render_learned_relaybp_params_selection(
    default: dict[str, Any], circuit_name: str, circuit_params: dict[str, Any]
) -> dict[str, Any]:
    all_run_dirs = discover_torch_run_dirs(
        circuit_name, circuit_params, "LearnedDMemBP"
    )
    with st.expander("LearnedRelayBP decoder params", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            run_dir = st.selectbox(
                "checkpoint",
                key="learned_relaybp_checkpoint",
                options=all_run_dirs,
                index=None,
                format_func=lambda x: x.name,
                help="Which checkpoint to load gamma0 vector from.",
            )
        with col2:
            gdi_low = st.number_input(
                "gamma_dist_interval low",
                key="learned_relaybp_gdi_low",
                value=default["gamma_dist_interval"][0],
                format="%f",
                help="Lower bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        with col3:
            gdi_high = st.number_input(
                "gamma_dist_interval high",
                key="learned_relaybp_gdi_high",
                value=default["gamma_dist_interval"][1],
                format="%f",
                help="Upper bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        if gdi_low >= gdi_high:
            st.error("gamma_dist_interval: low must be strictly less than high.")
            st.stop()
        col4, col5, col6 = st.columns(3)
        with col4:
            num_relays = st.number_input(
                "num_relays",
                key="learned_relaybp_num_relays",
                value=default["num_relays"],
                min_value=1,
                help="Number of DMemBP relays (beyond the initial stage).",
            )
        with col5:
            pre_iter = st.number_input(
                "pre_iter",
                key="learned_relaybp_pre_iter",
                value=default["pre_iter"],
                min_value=1,
                help="Number of iterations for the initial stage.",
            )
        with col6:
            max_iter_per_relay = st.number_input(
                "max_iter_per_relay",
                key="learned_relaybp_max_iter_per_relay",
                value=default["max_iter_per_relay"],
                min_value=1,
                help="Max number of iterations per DMemBP relay.",
            )
        col7, _, _ = st.columns(3)
        with col7:
            stop_nconv = st.number_input(
                "stop_nconv",
                key="learned_relaybp_stop_nconv",
                value=default["stop_nconv"],
                min_value=1,
                max_value=num_relays + 1,
                help="How many solutions to find before terminating.",
            )
    if run_dir is None:
        st.stop()
    ckpt_path = run_dir / "checkpoints" / "best_model.ckpt"
    ckpt_rel_path = ckpt_path.relative_to(REPO_ROOT).as_posix()
    return {
        "ckpt_rel_path": ckpt_rel_path,
        "gamma_dist_interval": (gdi_low, gdi_high),
        "num_relays": num_relays,
        "pre_iter": pre_iter,
        "max_iter_per_relay": max_iter_per_relay,
        "stop_nconv": stop_nconv,
    }


def _render_learned_multirelaybp_params_selection(
    default: dict[str, Any], circuit_name: str, circuit_params: dict[str, Any]
) -> dict[str, Any]:
    all_run_dirs = discover_torch_run_dirs(
        circuit_name, circuit_params, "LearnedDMemBP"
    )
    with st.expander("LearnedMultiRelayBP decoder params", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            run_dir = st.selectbox(
                "checkpoint",
                key="learned_multirelaybp_checkpoint",
                options=all_run_dirs,
                index=None,
                format_func=lambda x: x.name,
                help="Which checkpoint to load gamma0 vector from.",
            )
        with col2:
            gdi_low = st.number_input(
                "gamma_dist_interval low",
                key="learned_multirelaybp_gdi_low",
                value=default["gamma_dist_interval"][0],
                format="%f",
                help="Lower bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        with col3:
            gdi_high = st.number_input(
                "gamma_dist_interval high",
                key="learned_multirelaybp_gdi_high",
                value=default["gamma_dist_interval"][1],
                format="%f",
                help="Upper bound of the uniform distribution for random memory weights used in DMemBP relays.",
            )
        if gdi_low >= gdi_high:
            st.error("gamma_dist_interval: low must be strictly less than high.")
            st.stop()
        col4, col5, col6 = st.columns(3)
        with col4:
            num_chains = st.number_input(
                "num_chains",
                key="learned_multirelaybp_num_chains",
                value=default["num_chains"],
                min_value=1,
                help="Number of independent chains after the initial stage.",
            )
        with col5:
            num_relays = st.number_input(
                "num_relays",
                key="learned_multirelaybp_num_relays",
                value=default["num_relays"],
                min_value=1,
                help="Number of DMemBP relays in each chain (beyond the initial stage).",
            )
        with col6:
            pre_iter = st.number_input(
                "pre_iter",
                key="learned_multirelaybp_pre_iter",
                value=default["pre_iter"],
                min_value=1,
                help="Number of iterations for the initial stage.",
            )
        col7, col8, _ = st.columns(3)
        with col7:
            max_iter_per_relay = st.number_input(
                "max_iter_per_relay",
                key="learned_multirelaybp_max_iter_per_relay",
                value=default["max_iter_per_relay"],
                min_value=1,
                help="Max number of iterations per DMemBP relay.",
            )
        with col8:
            stop_nconv = st.number_input(
                "stop_nconv",
                key="learned_multirelaybp_stop_nconv",
                value=default["stop_nconv"],
                min_value=1,
                max_value=num_relays + 1,
                help="How many solutions to find before each chain terminates.",
            )
    if run_dir is None:
        st.stop()
    ckpt_path = run_dir / "checkpoints" / "best_model.ckpt"
    ckpt_rel_path = ckpt_path.relative_to(REPO_ROOT).as_posix()
    return {
        "ckpt_rel_path": ckpt_rel_path,
        "gamma_dist_interval": (gdi_low, gdi_high),
        "num_chains": num_chains,
        "num_relays": num_relays,
        "pre_iter": pre_iter,
        "max_iter_per_relay": max_iter_per_relay,
        "stop_nconv": stop_nconv,
    }


def render_task_selection() -> TaskMetadata:
    """Render benchmark task selection."""
    st.subheader("Select QEC circuit")

    col1, col2, col3 = st.columns(3)
    with col1:
        circuit_name = st.selectbox(
            "circuit name",
            key="circuit_name",
            options=list(DEFAULT_TASK_PARAMS.keys()),
            index=None,
        )
    if circuit_name is None:
        st.stop()

    with col2:
        selected_entry = st.selectbox(
            "circuit params",
            key="circuit_params",
            options=DEFAULT_TASK_PARAMS[circuit_name],
            index=None,
            format_func=lambda x: dict_to_str(x["circuit_params"]),
        )
    if selected_entry is None:
        st.stop()
    circuit_params = selected_entry.pop("circuit_params")
    default_error_rate = selected_entry.pop("error_rate")

    with col3:
        error_rate = st.number_input(
            "physical error rate",
            key="error_rate",
            value=default_error_rate,
            min_value=0.0,
            max_value=0.1,
            format="%f",
        )

    st.subheader("Select QEC decoder")

    col4, _, _ = st.columns(3)
    with col4:
        decoder_name = st.selectbox(
            "decoder name",
            key="decoder_name",
            options=list(selected_entry.keys()),
            index=None,
        )
    if decoder_name is None:
        st.stop()
    default_decoder_params = selected_entry[decoder_name]

    if decoder_name == "BP":
        decoder_params = _render_bp_params_selection(default_decoder_params)
    elif decoder_name == "BPOSD":
        decoder_params = _render_bposd_params_selection(default_decoder_params)
    elif decoder_name == "MemBP":
        decoder_params = _render_membp_params_selection(default_decoder_params)
    elif decoder_name == "RelayBP":
        decoder_params = _render_relaybp_params_selection(default_decoder_params)
    elif decoder_name == "MultiRelayBP":
        decoder_params = _render_multirelaybp_params_selection(default_decoder_params)
    elif decoder_name == "LearnedDMemBP":
        decoder_params = _render_learned_dmembp_params_selection(
            default_decoder_params, circuit_name, circuit_params
        )
    elif decoder_name == "LearnedRelayBP":
        decoder_params = _render_learned_relaybp_params_selection(
            default_decoder_params, circuit_name, circuit_params
        )
    elif decoder_name == "LearnedMultiRelayBP":
        decoder_params = _render_learned_multirelaybp_params_selection(
            default_decoder_params, circuit_name, circuit_params
        )
    elif decoder_name in ["MWPM", "UnionFind"]:
        decoder_params = {}
    else:
        raise NotImplementedError(
            f"{decoder_name} decoder is not yet supported by the benchmark suite."
        )

    return TaskMetadata(
        circuit_name=circuit_name,
        circuit_params=circuit_params,
        error_rate=error_rate,
        decoder_name=decoder_name,
        decoder_params=decoder_params,
    )

def check_benchmark_completeness(
    task_metadata: TaskMetadata,
    csv_path: Path,
    shots_cap: int,
    errors_cap: int,
) -> bool:
    stats_list = TaskStats.load_csv(csv_path)
    stats = TaskStats.find_by_metadata(stats_list, task_metadata)
    if stats is None:
        st.info(
            "No benchmark results found for the selected task. Please run the benchmark."
        )
        return False
    elif not stats.is_complete(shots_cap, errors_cap):
        st.info(
            "The selected benchmark task is not complete:\n\n"
            f"Shots: {stats.shots:_}/{shots_cap:_}\n\n"
            f"Errors: {stats.obser_errors}/{errors_cap}\n\n"
            "Please resume the benchmark."
        )
        return False
    else:
        st.success(
            "Benchmark complete!\n\n"
            f"Shots: {stats.shots:_}/{shots_cap:_}\n\n"
            f"Errors: {stats.obser_errors}/{errors_cap}\n\n"
            f"Failure Rate: {stats.failure_rate:.2e}"
        )
        return True
