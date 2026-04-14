"""Shared Streamlit UI components used by all benchmark pages."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
import streamlit as st
from omegaconf import OmegaConf

from bench.constants import BASELINE_DECODERS_GRAPHLIKE, BASELINE_DECODERS_HYPERGRAPH
from bench.params import QECParams
from constants import (
    CIRCUITS_ROOT,
    DEFAULT_BP_MAX_ITER,
    DEFAULT_MEMBP_MAX_ITER,
    DEFAULT_MEMBP_GAMMA,
    DEFAULT_RELAYBP_PRE_ITER,
    DEFAULT_RELAYBP_MAX_ITER_PER_RELAY,
    DEFAULT_RELAYBP_NUM_RELAYS,
    DEFAULT_RELAYBP_STOP_NCONV,
    DEFAULT_RELAYBP_GAMMA0,
    DEFAULT_RELAYBP_GDI,
    DEFAULT_BPOSD_MAX_ITER,
    DEFAULT_BPOSD_OSD_ORDER,
    DEFAULT_PYTORCH_MAX_ITER,
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


def render_baselines_selection(
    qec_params: QECParams,
) -> tuple[list[str], dict[str, dict]]:
    """Render baseline decoder selection and per-decoder config panels in the main area.

    Return ``(selected_decoders, baseline_decoder_params)`` where
    ``baseline_decoder_params`` maps each selected decoder name to its config dict.
    """
    st.subheader("Select baseline decoder(s)")
    if qec_params.code == "RotatedSurfaceCode":
        available_decoders = BASELINE_DECODERS_GRAPHLIKE
    elif qec_params.code.startswith("BB_"):
        available_decoders = BASELINE_DECODERS_HYPERGRAPH
    else:
        raise ValueError(f"Unknown code: {qec_params.code}")

    selected_decoders = st.multiselect(
        "Baseline decoder(s) to benchmark against",
        options=available_decoders,
        default=available_decoders,
    )

    baseline_decoder_params: dict[str, dict] = {}
    for name in selected_decoders:
        if name == "BP":
            with st.expander("BP configuration", expanded=True):
                col1, _, _ = st.columns(3)
                with col1:
                    max_iter = st.number_input(
                        "max_iter",
                        value=DEFAULT_BP_MAX_ITER,
                        min_value=1,
                        key="bp_max_iter",
                        help="Max number of BP iterations.",
                    )
            baseline_decoder_params["BP"] = {"max_iter": max_iter}
        elif name == "MemBP":
            with st.expander("MemBP configuration", expanded=True):
                col1, col2, _ = st.columns(3)
                with col1:
                    max_iter = st.number_input(
                        "max_iter",
                        value=DEFAULT_MEMBP_MAX_ITER,
                        min_value=1,
                        key="membp_max_iter",
                        help="Max number of BP iterations.",
                    )
                with col2:
                    gamma = st.number_input(
                        "gamma",
                        value=DEFAULT_MEMBP_GAMMA.get(qec_params, 0.0),
                        key="membp_gamma",
                        help="Memory coefficient.",
                    )
            baseline_decoder_params["MemBP"] = {"max_iter": max_iter, "gamma": gamma}
        elif name == "RelayBP":
            with st.expander("RelayBP configuration", expanded=True):
                default_gdi = DEFAULT_RELAYBP_GDI.get(qec_params, (-0.5, 0.5))
                col1, col2, col3 = st.columns(3)
                with col1:
                    gamma0 = st.number_input(
                        "gamma0",
                        value=DEFAULT_RELAYBP_GAMMA0.get(qec_params, 0.0),
                        format="%f",
                        step=0.0001,
                        key="relaybp_gamma0",
                        help="Memory parameter for the first MemBP instance.",
                    )
                with col2:
                    gdi_low = st.number_input(
                        "gamma_dist_interval low",
                        value=default_gdi[0],
                        format="%f",
                        step=0.0001,
                        key="relaybp_gdi_low",
                        help="Lower bound of the uniform distribution for random memory weights used in DMemBP relays.",
                    )
                with col3:
                    gdi_high = st.number_input(
                        "gamma_dist_interval high",
                        value=default_gdi[1],
                        format="%f",
                        step=0.0001,
                        key="relaybp_gdi_high",
                        help="Upper bound of the uniform distribution for random memory weights used in DMemBP relays.",
                    )
                if gdi_low >= gdi_high:
                    st.error(
                        "gamma_dist_interval: low must be strictly less than high."
                    )
                    st.stop()
                col4, col5, col6 = st.columns(3)
                with col4:
                    num_relays = st.number_input(
                        "num_relays",
                        value=DEFAULT_RELAYBP_NUM_RELAYS,
                        min_value=1,
                        key="relaybp_num_relays",
                        help="Number of DMemBP relays (beyond the first MemBP instance).",
                    )
                with col5:
                    pre_iter = st.number_input(
                        "pre_iter",
                        value=DEFAULT_RELAYBP_PRE_ITER,
                        min_value=1,
                        key="relaybp_pre_iter",
                        help="Number of iterations for the first MemBP instance.",
                    )
                with col6:
                    max_iter_per_relay = st.number_input(
                        "max_iter_per_relay",
                        value=DEFAULT_RELAYBP_MAX_ITER_PER_RELAY,
                        min_value=1,
                        key="relaybp_max_iter_per_relay",
                        help="Max number of iterations per DMemBP relay.",
                    )
                col7, _, _ = st.columns(3)
                with col7:
                    stop_nconv = st.number_input(
                        "stop_nconv",
                        value=DEFAULT_RELAYBP_STOP_NCONV,
                        min_value=1,
                        key="relaybp_stop_nconv",
                        help="How many solutions to find before terminating.",
                    )
            baseline_decoder_params["RelayBP"] = {
                "gamma0": gamma0,
                "gamma_dist_interval": [gdi_low, gdi_high],
                "num_relays": num_relays,
                "pre_iter": pre_iter,
                "max_iter_per_relay": max_iter_per_relay,
                "stop_nconv": stop_nconv,
            }
        elif name == "BPOSD":
            with st.expander("BPOSD configuration", expanded=True):
                col1, col2, _ = st.columns(3)
                with col1:
                    max_iter = st.number_input(
                        "max_iter",
                        value=DEFAULT_BPOSD_MAX_ITER,
                        min_value=1,
                        key="bposd_max_iter",
                        help="Max number of BP iterations.",
                    )
                with col2:
                    osd_order = st.number_input(
                        "osd_order",
                        value=DEFAULT_BPOSD_OSD_ORDER,
                        min_value=0,
                        key="bposd_osd_order",
                        help="OSD order.",
                    )
                baseline_decoder_params["BPOSD"] = {
                    "max_bp_iter": max_iter,
                    "osd_method": "OSD_CS",
                    "osd_order": osd_order,
                }
        else:
            baseline_decoder_params[name] = {}

    return selected_decoders, baseline_decoder_params


def render_p_list_selection() -> list[float]:
    """Render physical error rates selection in the sidebar.

    Return ``p_list``. Call ``st.stop()`` if the user input ``p_list`` is empty or unparsable.
    """
    with st.sidebar:
        st.subheader("Select benchmark settings")
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
    return p_list


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


def render_torchdecoder_selection(
    run_dirs: list[Path], qec_params: QECParams
) -> tuple[list[Path], dict[str, Any]]:
    """Render torch decoder tables with row selection and individual config expanders,
    as well as a shared configuration panel for the PyTorch decoder(s).

    Return ``(selected_run_dirs, torchdecoder_shared_params)``.

    ``qec_params`` is used to namespace widget keys so that changing the
    upstream QEC selection resets row selections instead of carrying over
    stale indices from a previous table.
    """
    qec_key = (
        f"{qec_params.code}_{qec_params.noise_model}"
        f"_{qec_params.d}_{qec_params.rounds}_{qec_params.basis}"
    )
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
            key=f"pytorch_decoder_selection_{qec_key}_{model_name}",
            on_select="rerun",
            selection_mode="multi-row",
            hide_index=True,
        )
        selected_indices = event.selection.rows or []  # type: ignore
        selected_run_dirs.extend([group[i] for i in selected_indices])

        with st.expander(f"View full configs of {model_name} used for training"):
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

    with st.expander("PyTorch decoder configuration", expanded=True):
        col1, col2, _ = st.columns(3)
        with col1:
            which_prior = st.selectbox(
                "prior",
                options=[
                    "Load from model checkpoint",
                    "Derive from physical error rate",
                ],
                index=0,
                help="Which prior vector to use for decoding.",
            )
            use_prior_in_ckpt = which_prior == "Load from model checkpoint"
        with col2:
            max_iter = st.number_input(
                "max_iter",
                value=DEFAULT_PYTORCH_MAX_ITER,
                min_value=1,
                key="pytorch_max_iter",
                help="Max number of decoding iterations for inference.",
            )

    torchdecoder_shared_params = {
        "use_prior_in_ckpt": use_prior_in_ckpt,
        "max_iter": max_iter,
    }
    return selected_run_dirs, torchdecoder_shared_params


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


def render_circuit_source_selection() -> str:
    """Render circuit source toggle in the main area.

    Return ``"generate"`` or ``"stim_file"``.
    """
    source = st.radio(
        "Circuit source",
        options=["Generate from code parameters", "Load from circuit files"],
        horizontal=True,
    )
    return "generate" if source == "Generate from code parameters" else "stim_file"


def _parse_d_rounds_basis_dirname(dirname: str) -> tuple[int, int, str]:
    """Parse d, rounds, basis from a subdirectory name.

    E.g. ``"d=6_rounds=6_basis=Z"`` -> ``(6, 6, "Z")``.
    """
    params = {}
    for token in dirname.split("_"):
        key, _, value = token.partition("=")
        params[key] = value
    return int(params["d"]), int(params["rounds"]), params["basis"]


def render_stim_file_selection(
    p_list: list[float],
) -> tuple[QECParams, dict[float, Path]]:
    """Render stim circuit file selection UI.

    Scans the ``circuits/`` directory for code families and d/rounds/basis
    subfolders. Validates that a ``.stim`` file exists for each error rate
    in ``p_list``.

    Return ``(qec_params, stim_file_paths)`` where ``stim_file_paths``
    maps each ``p`` to its ``.stim`` file path.
    """
    st.subheader("Select circuit files")

    if not CIRCUITS_ROOT.is_dir():
        st.warning(f"Circuits directory not found: `{CIRCUITS_ROOT}`")
        st.stop()

    code_noise_dirnames = sorted(
        d.name
        for d in CIRCUITS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not code_noise_dirnames:
        st.warning(
            "No (code, noise model) subdirectories found in the circuits directory."
        )
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        selected_code_noise_dirname = st.selectbox(
            "code, noise model",
            options=code_noise_dirnames,
            index=None,
            format_func=lambda x: ", ".join(x.rsplit("_", 1)),
        )
    if selected_code_noise_dirname is None:
        st.stop()

    code, noise_model = selected_code_noise_dirname.rsplit("_", 1)
    subdir = CIRCUITS_ROOT / selected_code_noise_dirname

    d_rounds_basis_dirnames = sorted(
        d.name for d in subdir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not d_rounds_basis_dirnames:
        st.warning("No (d, rounds, basis) subdirectories found.")
        st.stop()

    with col2:
        selected_d_rounds_basis_dirname = st.selectbox(
            "d, rounds, basis",
            options=d_rounds_basis_dirnames,
            index=None,
            format_func=lambda x: x.replace("_", ", "),
        )
    if selected_d_rounds_basis_dirname is None:
        st.stop()

    d, rounds, basis = _parse_d_rounds_basis_dirname(selected_d_rounds_basis_dirname)
    subdir = subdir / selected_d_rounds_basis_dirname

    # Map each p to its .stim file, validating existence
    stim_file_paths: dict[float, Path] = {}
    missing: list[float] = []
    available_files = {f.stem: f for f in subdir.glob("*.stim")}
    for p in p_list:
        key = f"error_rate={p}"
        if key in available_files:
            stim_file_paths[p] = available_files[key]
        else:
            missing.append(p)

    if missing:
        available_rates = sorted(f.stem.split("=", 1)[1] for f in subdir.glob("*.stim"))
        st.error(
            f"No circuit file found for error rate(s): {missing}. "
            f"Available error rates: {available_rates}"
        )
        st.stop()

    qec_params = QECParams(
        code=code,
        noise_model=noise_model,
        d=d,
        rounds=rounds,
        basis=basis,
    )
    return qec_params, stim_file_paths


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
