"""Streamlit UI components for the benchmark app."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from bench.constants import (
    BASELINE_DECODERS_GRAPHLIKE,
    BASELINE_DECODERS_HYPERGRAPH,
    DEFAULT_BASELINE_DECODERS_GRAPHLIKE,
    DEFAULT_BASELINE_DECODERS_HYPERGRAPH,
)
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
    DEFAULT_ENS_SERIAL_BP_MAX_ITER,
    DEFAULT_ENS_SERIAL_BP_ENSEMBLE_SIZE,
    DEFAULT_ENS_SERIAL_BP_TOPK,
    DEFAULT_ENS_SERIAL_BP_SEED,
    DEFAULT_PYTORCH_MAX_ITER,
    DEFAULT_P_LIST,
    ALL_CODE_NOISE_PAIRS,
    CODE_NOISE_PAIR_TO_D_ROUNDS_BASIS_TRIPLES,
)
from torchdecoder_utils import (
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
        default_decoders = DEFAULT_BASELINE_DECODERS_GRAPHLIKE
    elif qec_params.code.startswith("BB_") or qec_params.code == "HexColorCode":
        available_decoders = BASELINE_DECODERS_HYPERGRAPH
        default_decoders = DEFAULT_BASELINE_DECODERS_HYPERGRAPH
    else:
        raise ValueError(f"Unknown code: {qec_params.code}")

    selected_decoders = st.multiselect(
        "Baseline decoder(s) to benchmark against",
        options=available_decoders,
        default=default_decoders,
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
                        max_value=num_relays + 1,
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
        elif name == "EnsSerialBP":
            with st.expander("EnsSerialBP configuration", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_iter = st.number_input(
                        "max_iter",
                        value=DEFAULT_ENS_SERIAL_BP_MAX_ITER,
                        min_value=1,
                        key="ens_serial_bp_max_iter",
                        help="Max number of iterations (one iteration = one full pass of all VNs).",
                    )
                with col2:
                    ensemble_size = st.number_input(
                        "ensemble_size",
                        value=DEFAULT_ENS_SERIAL_BP_ENSEMBLE_SIZE,
                        min_value=1,
                        key="ens_serial_bp_ensemble_size",
                        help="Number of serial-schedule BP members run in parallel.",
                    )
                with col3:
                    topk = st.number_input(
                        "topk",
                        value=min(DEFAULT_ENS_SERIAL_BP_TOPK, ensemble_size),
                        min_value=1,
                        max_value=ensemble_size,
                        key="ens_serial_bp_topk",
                        help="Stop once this many members converge.",
                    )
                col4, _, _ = st.columns(3)
                with col4:
                    seed = st.number_input(
                        "random seed",
                        value=DEFAULT_ENS_SERIAL_BP_SEED,
                        key="ens_serial_bp_seed",
                        help="Random seed for generating permutations of VNs.",
                    )
            baseline_decoder_params["EnsSerialBP"] = {
                "max_iter": max_iter,
                "ensemble_size": ensemble_size,
                "topk": topk,
                "seed": seed,
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


def render_qec_selection() -> QECParams:
    """Render QEC parameter selection."""
    st.subheader("Select QEC parameters")

    col1, col2 = st.columns(2)
    with col1:
        selected_code_noise_pair = st.selectbox(
            "code, noise model",
            options=ALL_CODE_NOISE_PAIRS,
            index=None,
            format_func=lambda x: f"{x[0]}, {x[1]}",
        )
    if selected_code_noise_pair is None:
        st.stop()
    code, noise_model = selected_code_noise_pair

    with col2:
        selected_d_rounds_basis_triple = st.selectbox(
            "d, rounds, basis",
            options=CODE_NOISE_PAIR_TO_D_ROUNDS_BASIS_TRIPLES[selected_code_noise_pair],
            index=None,
            format_func=lambda x: f"{x[0]}, {x[1]}, {x[2]}",
        )
    if selected_d_rounds_basis_triple is None:
        st.stop()
    d, rounds, basis = selected_d_rounds_basis_triple

    return QECParams(
        code=code,
        noise_model=noise_model,
        d=d,
        rounds=rounds,
        basis=basis,
    )


def render_torchdecoder_selection(
    run_dirs: list[Path],
    qec_params: QECParams,
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
        st.caption(
            "When device is CPU and the selected model is LearnedDMemBP, "
            "inference runs via the equivalent Rust DMemBPDecoder for speed."
        )
        relaybp_mode_pending = st.session_state.get("td_relaybp_mode", False)
        col1, col2, col3 = st.columns(3)
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
                disabled=relaybp_mode_pending,
                help=(
                    "Max number of decoding iterations for inference. "
                    "Ignored when 'Run as RelayBP' is enabled; the effective "
                    "budget becomes pre_iter + num_relays * max_iter_per_relay."
                ),
            )
        with col3:
            relaybp_mode = st.checkbox(
                "Run as RelayBP",
                value=False,
                key="td_relaybp_mode",
                help=(
                    "If on, the CPU swap builds a Rust RelayBPDecoder using "
                    "the checkpoint's gamma vector as gamma0 instead of "
                    "running DMemBP. Requires device=CPU and LearnedDMemBP."
                ),
            )

        relaybp_params: dict[str, Any] = {}
        if relaybp_mode:
            st.markdown("**RelayBP hyperparameters**")
            default_gdi = DEFAULT_RELAYBP_GDI.get(qec_params, (-0.5, 0.5))
            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                gdi_low = st.number_input(
                    "gamma_dist_interval low",
                    value=default_gdi[0],
                    format="%f",
                    step=0.0001,
                    key="td_relaybp_gdi_low",
                    help="Lower bound of the uniform distribution for random gamma vectors at each relay stage.",
                )
            with rcol2:
                gdi_high = st.number_input(
                    "gamma_dist_interval high",
                    value=default_gdi[1],
                    format="%f",
                    step=0.0001,
                    key="td_relaybp_gdi_high",
                    help="Upper bound of the uniform distribution for random gamma vectors at each relay stage.",
                )
            with rcol3:
                num_relays = st.number_input(
                    "num_relays",
                    value=DEFAULT_RELAYBP_NUM_RELAYS,
                    min_value=1,
                    key="td_relaybp_num_relays",
                    help="Number of DMemBP relays beyond the first stage.",
                )
            if gdi_low >= gdi_high:
                st.error("gamma_dist_interval: low must be strictly less than high.")
                st.stop()
            rcol4, rcol5, rcol6 = st.columns(3)
            with rcol4:
                pre_iter = st.number_input(
                    "pre_iter",
                    value=DEFAULT_RELAYBP_PRE_ITER,
                    min_value=1,
                    key="td_relaybp_pre_iter",
                    help="Max iterations for the first DMemBP stage (uses checkpoint gamma as gamma0).",
                )
            with rcol5:
                max_iter_per_relay = st.number_input(
                    "max_iter_per_relay",
                    value=DEFAULT_RELAYBP_MAX_ITER_PER_RELAY,
                    min_value=1,
                    key="td_relaybp_max_iter_per_relay",
                    help="Max iterations per relay stage.",
                )
            with rcol6:
                stop_nconv = st.number_input(
                    "stop_nconv",
                    value=DEFAULT_RELAYBP_STOP_NCONV,
                    min_value=1,
                    max_value=num_relays + 1,
                    key="td_relaybp_stop_nconv",
                    help="Stop after collecting this many converged candidates.",
                )
            relaybp_params = {
                "gamma_dist_interval": [gdi_low, gdi_high],
                "num_relays": num_relays,
                "pre_iter": pre_iter,
                "max_iter_per_relay": max_iter_per_relay,
                "stop_nconv": stop_nconv,
            }

    torchdecoder_shared_params: dict[str, Any] = {
        "use_prior_in_ckpt": use_prior_in_ckpt,
        "relaybp_mode": relaybp_mode,
    }
    if relaybp_mode:
        torchdecoder_shared_params["relaybp"] = relaybp_params
    else:
        torchdecoder_shared_params["max_iter"] = max_iter
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


def validate_stim_files(
    qec_params: QECParams,
    p_list: list[float],
) -> None:
    """Validate that stim circuit files exist. Call `st.stop()` if not."""
    if not CIRCUITS_ROOT.is_dir():
        st.warning(f"Circuits directory not found: `{CIRCUITS_ROOT}`")
        st.stop()

    subdir = (
        CIRCUITS_ROOT
        / f"{qec_params.code}_{qec_params.noise_model}"
        / f"d={qec_params.d}_rounds={qec_params.rounds}_basis={qec_params.basis}"
    )
    available_rates = sorted(
        float(f.stem.split("=", 1)[1]) for f in subdir.glob("*.stim")
    )
    missing_rates = [p for p in p_list if p not in available_rates]

    if missing_rates:
        st.error(
            f"No circuit file found for error rate(s): {missing_rates}. "
            f"Available error rates: {available_rates}"
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
