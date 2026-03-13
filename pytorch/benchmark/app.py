"""
Streamlit app for Monte Carlo benchmarking of PyTorch decoders.
Run with: `streamlit run pytorch/benchmark/app.py`
"""
from io import BytesIO
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import sinter
import torch
import streamlit as st
from omegaconf import OmegaConf

from utils import (
    RUNS_ROOT,
    BENCHMARK_CSV_FILENAME,
    is_unique,
    load_run_config,
    extract_pytorch_decoder_name,
    extract_pytorch_decoder_run_id,
    flatten_config,
    get_differing_keys,
)
from benchmark import run_benchmark
from baselines_benchmark import (
    BASELINE_DECODERS,
    get_baseline_csv_path,
)


DEFAULT_P_LIST = [0.004, 0.006, 0.008, 0.01, 0.012]
DEFAULT_MAX_SHOTS = 100_000_000
DEFAULT_MAX_ERRORS = 100
DEFAULT_MAX_ITER = 50


def discover_run_dirs() -> list[Path]:
    """
    Discover all run_dirs: These are the subdirectories of pytorch/runs that contain a checkpoints/best_model.ckpt file.
    """
    run_dirs: list[Path] = []
    for p in RUNS_ROOT.rglob("checkpoints/best_model.ckpt"):
        run_dirs.append(p.parent.parent)
    return run_dirs


def group_run_dirs_by_code_and_noise(run_dirs: list[Path]) -> defaultdict[tuple[str, str], list[Path]]:
    """
    Split run_dirs into groups according to (code, noise_model) pairs.
    """
    grouped = defaultdict[tuple[str, str], list[Path]](list)
    for run_dir in run_dirs:
        cfg = load_run_config(run_dir)
        grouped[(cfg.qec.code, cfg.qec.noise_model)].append(run_dir)
    return grouped


def group_run_dirs_by_d_rounds_basis(run_dirs: list[Path]) -> defaultdict[tuple[int, int, str], list[Path]]:
    """
    Split run_dirs into groups according to (d, rounds, basis) triples.
    """
    grouped = defaultdict[tuple[int, int, str], list[Path]](list)
    for run_dir in run_dirs:
        cfg = load_run_config(run_dir)
        grouped[(cfg.qec.d, cfg.qec.rounds, cfg.qec.basis)].append(run_dir)
    return grouped


def group_run_dirs_by_decoder_model_name(run_dirs: list[Path]) -> defaultdict[str, list[Path]]:
    """
    Split run_dirs into groups according to decoder model name.
    """
    grouped = defaultdict[str, list[Path]](list)
    for run_dir in run_dirs:
        cfg = load_run_config(run_dir)
        grouped[cfg.model.name].append(run_dir)
    return grouped


def filter_stats(
    stats: list[sinter.TaskStats],
    *,
    p_list: list[float],
    max_iter: int,
    max_shots: int,
    max_errors: int,
) -> list[sinter.TaskStats]:
    """
    Filter the list of `sinter.TaskStats` to only include those elements `s` such that:
    - `s.json_metadata["p"]` is in `p_list`
    - `s.json_metadata["max_iter"] == max_iter` if the key "max_iter" exists in `s.json_metadata`
    - either `s.shots >= max_shots` or `s.errors >= max_errors`
    """
    filtered: list[sinter.TaskStats] = []
    for s in stats:
        if s.json_metadata["p"] not in p_list:
            continue
        if "max_iter" in s.json_metadata and s.json_metadata["max_iter"] != max_iter:
            continue
        if s.shots < max_shots and s.errors < max_errors:
            continue
        filtered.append(s)
    return filtered


def load_and_merge_stats(
    *,
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
    run_dirs: list[Path],
    baseline_decoders: list[str],
    max_iter: int,
    p_list: list[float],
    max_shots: int,
    max_errors: int,
) -> tuple[list[sinter.TaskStats], list[Path], list[str]]:
    """
    Load and merge PyTorch decoders' stats and baseline decoders' stats into a single list.

    Return `(all_stats, missing_run_dirs, missing_baseline_decoders)` where:
    - `all_stats` is a merged list of `sinter.TaskStats`.
    - `missing_run_dirs` is a sublist of `run_dirs` with missing or incomplete data.
    - `missing_baseline_decoders` is a sublist of `baseline_decoders` with missing or incomplete data.
    """
    all_stats: list[sinter.TaskStats] = []
    missing_run_dirs: list[Path] = []
    missing_baseline_decoders: list[str] = []

    # Load PyTorch decoders' stats.
    for run_dir in run_dirs:
        csv_path = run_dir / BENCHMARK_CSV_FILENAME
        if not csv_path.exists():
            missing_run_dirs.append(run_dir)
            continue
        stats = sinter.read_stats_from_csv_files(csv_path)
        stats = filter_stats(
            stats,
            p_list=p_list,
            max_iter=max_iter,
            max_shots=max_shots,
            max_errors=max_errors,
        )
        if set(s.json_metadata["p"] for s in stats) != set(p_list):
            missing_run_dirs.append(run_dir)
            continue
        all_stats.extend(stats)

    # Load baseline decoders' stats.
    for decoder in baseline_decoders:
        csv_path = get_baseline_csv_path(
            code, noise_model, d, rounds, basis, decoder
        )
        if not csv_path.exists():
            missing_baseline_decoders.append(decoder)
            continue
        stats = sinter.read_stats_from_csv_files(csv_path)
        stats = filter_stats(
            stats,
            p_list=p_list,
            max_iter=max_iter,
            max_shots=max_shots,
            max_errors=max_errors,
        )
        if set(s.json_metadata["p"] for s in stats) != set(p_list):
            missing_baseline_decoders.append(decoder)
            continue
        all_stats.extend(stats)

    return all_stats, missing_run_dirs, missing_baseline_decoders


def main():
    st.set_page_config(page_title="Decoder Benchmark", layout="wide", page_icon="📈")
    st.title("Monte Carlo Benchmark")

    # Discover all run_dirs
    run_dirs = discover_run_dirs()
    if len(run_dirs) == 0:
        st.warning("No trained PyTorch decoders found. Train a model first.")
        st.stop()
    if not is_unique(run_dirs):
        raise Exception("Duplicate run_dirs found.")

    # Sidebar: baseline decoders and benchmark params
    with st.sidebar:
        st.subheader("Select baseline decoder(s)")
        selected_baseline_decoders = st.multiselect(
            "Baseline decoder(s) to benchmark against",
            options=BASELINE_DECODERS,
            default=BASELINE_DECODERS,
        )

        st.subheader("Select benchmark parameters")
        max_iter = st.number_input(
            "Max number of decoding iterations",
            value=DEFAULT_MAX_ITER,
            min_value=1,
            help="Only apply to iterative decoders (e.g., BP, LearnedDMemBP)."
        )
        p_list_str = st.text_input(
            "Physical error rates (comma-separated)",
            value=", ".join(str(p) for p in DEFAULT_P_LIST),
            help="List of physical error rates to benchmark at, separated by commas."
        )
        try:
            p_list = [float(x.strip()) for x in p_list_str.split(",") if x.strip()]
        except ValueError:
            st.error("Cannot parse into a list of floats.")
            st.stop()
        if len(p_list) == 0:
            st.warning("Please enter at least one physical error rate.")
            st.stop()
        max_shots = st.number_input(
            "Max number of shots",
            value=DEFAULT_MAX_SHOTS,
            min_value=1,
            help="Stop Monte Carlo sampling after taking this many shots."
        )
        max_errors = st.number_input(
            "Max number of failures",
            value=DEFAULT_MAX_ERRORS,
            min_value=1,
            help="Stop Monte Carlo sampling after having seen this many failures."
        )
        device = st.selectbox(
            "Device",
            options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            index=0,
            help="Device to use for benchmarking PyTorch decoders.",
        )
        if device == "cuda":
            default_num_workers = 1
        else:
            default_num_workers = max(1, (os.cpu_count() or 1) - 1)
        num_workers = st.number_input(
            "Number of workers",
            value=default_num_workers,
            min_value=1,
            help="Number of workers used for benchmarking.",
        )
        if device == "cuda" and num_workers > 1:
            st.warning("It is not recommended to use more than 1 worker for benchmarking on GPU.")

    # Main: Select QEC parameters
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
    run_dirs = grouped[selected_code_noise_pair]  # Filter run_dirs by code and noise model

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
    run_dirs = grouped[selected_d_rounds_basis_triple]  # Filter run_dirs by d, rounds, and basis

    # Main: Tables of PyTorch decoders (one per model type) with row selection; compare configs when 2+ runs per model
    st.subheader("Select PyTorch decoder(s)")
    st.caption("One table per decoder model. "
               "Only differing config fields are shown in each table. "
               "Click the checkboxes to select runs to benchmark. "
               "Click the expander below to view full configs.")
    grouped = group_run_dirs_by_decoder_model_name(run_dirs)
    selected_run_dirs: list[Path] = []
    for model_name in sorted(grouped.keys()):
        group = sorted(grouped[model_name], key=extract_pytorch_decoder_run_id)
        df_data = {"Run ID": [r.name for r in group]}
        if len(group) >= 2:
            configs = [load_run_config(r) for r in group]
            flat_configs = [flatten_config(OmegaConf.to_container(cfg, resolve=True)) for cfg in configs]
            diff_keys = get_differing_keys(flat_configs)
            df_data.update({
                k: [str(c.get(k, "N/A")) for c in flat_configs]
                for k in sorted(diff_keys)
            })
        df = pd.DataFrame(df_data)

        st.markdown(f"**{model_name}**")
        event = st.dataframe(
            df,
            width="stretch",
            key=f"pytorch_decoder_selection_{code}_{noise_model}_{d}_{rounds}_{basis}_{model_name}",
            on_select="rerun",
            selection_mode="multi-row",
            hide_index=True,
        )
        selected_indices = event.selection.rows or []
        selected_run_dirs.extend([group[i] for i in selected_indices])

        # View configs expander
        with st.expander("View full configs"):
            selected_view_config = st.selectbox(
                f"{model_name} config",
                options=group,
                index=None,
                format_func=lambda r: r.name,
                key=f"config_viewer_{code}_{noise_model}_{d}_{rounds}_{basis}_{model_name}",
                placeholder="Choose a run ID",
            )
            if selected_view_config:
                cfg = load_run_config(selected_view_config)
                cfg_yaml = OmegaConf.to_yaml(cfg)
                st.code(cfg_yaml, language="yaml")

    # Early exit when no decoders selected
    if len(selected_run_dirs) == 0 and len(selected_baseline_decoders) == 0:
        st.warning("Please select at least one PyTorch decoder or baseline decoder to benchmark.")
        st.stop()

    stats, missing_run_dirs, missing_baseline_decoders = load_and_merge_stats(
        code=code,
        noise_model=noise_model,
        d=d,
        rounds=rounds,
        basis=basis,
        run_dirs=selected_run_dirs,
        baseline_decoders=selected_baseline_decoders,
        max_iter=max_iter,
        p_list=p_list,
        max_shots=max_shots,
        max_errors=max_errors,
    )

    # If data is incomplete, show warning and "Run benchmark" button
    if len(missing_run_dirs) > 0 or len(missing_baseline_decoders) > 0:
        missing_list = [extract_pytorch_decoder_name(r) for r in missing_run_dirs] + missing_baseline_decoders
        missing_list_str = "\n\n".join(f"• {d}" for d in missing_list)
        st.warning(
            "The following decoders have missing or incomplete benchmark data:\n\n"
            f"{missing_list_str}\n\n"
            "Please run benchmark first."
        )
        if st.button("Run benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                run_benchmark(
                    code=code,
                    noise_model=noise_model,
                    d=d,
                    rounds=rounds,
                    basis=basis,
                    run_dirs=missing_run_dirs,
                    baseline_decoders=missing_baseline_decoders,
                    max_iter=max_iter,
                    p_list=p_list,
                    max_shots=max_shots,
                    max_errors=max_errors,
                    num_workers=num_workers,
                    device=device,
                )
                st.success("Benchmark completed successfully.")
                st.rerun()
        st.stop()

    # Data is complete, show plots
    st.subheader("Logical Error Rate (LER) vs Physical Error Rate (PER)")
    ler_mode = st.radio(
        "LER calculation method",
        options=["per shot", "per round"],
        horizontal=True,
    )

    def filter_func(stat: sinter.TaskStats) -> bool:
        cond1 = stat.json_metadata["p"] in p_list
        if "max_iter" in stat.json_metadata:
            cond2 = stat.json_metadata["max_iter"] == max_iter
        else:
            cond2 = True
        return cond1 and cond2

    def group_func(stat: sinter.TaskStats) -> dict:
        decoder = stat.json_metadata["decoder"]
        if decoder in BASELINE_DECODERS:
            return {
                "label": decoder,
                "linestyle": "dashed",
            }
        else:
            return {
                "label": decoder,
                "linestyle": "solid",
            }

    fig, ax = plt.subplots(1, 1)
    if ler_mode == "per shot":
        sinter.plot_error_rate(
            ax=ax,
            stats=stats,
            x_func=lambda stat: stat.json_metadata["p"],
            filter_func=filter_func,
            group_func=group_func,
            plot_args_func=lambda index, group_key: {
                "linestyle": group_key["linestyle"],
            },
        )
        ax.set_ylabel("LER per shot")
    else:  # ler_mode == "per round"
        sinter.plot_error_rate(
            ax=ax,
            stats=stats,
            x_func=lambda stat: stat.json_metadata["p"],
            filter_func=filter_func,
            group_func=group_func,
            failure_units_per_shot_func=lambda stat: stat.json_metadata["rounds"],
            plot_args_func=lambda index, group_key: {
                "linestyle": group_key["linestyle"],
            },
        )
        ax.set_ylabel("LER per round")
    ax.loglog()
    ax.grid(axis='y')
    ax.set_title(f"{code}, {noise_model}, d={d}, rounds={rounds}, basis={basis}")
    ax.set_xlabel("PER")
    ax.legend()
    # Constrain plot width by placing it in a centered column (avoids full-width stretch)
    _, plot_col, _ = st.columns([1, 3, 1])
    with plot_col:
        st.pyplot(fig, width="stretch")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    st.download_button(
        "Download plot as PNG",
        data=buf.getvalue(),
        file_name="benchmark_LER_vs_PER.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
