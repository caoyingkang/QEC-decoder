"""
Streamlit app for Monte Carlo benchmarking of PyTorch decoders.
Run with: `streamlit run pytorch/benchmark/app.py`
"""
from io import BytesIO
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import sinter
import streamlit as st

from utils import (
    RUNS_ROOT,
    BENCHMARK_CSV_FILENAME,
    is_unique,
    load_run_config,
    extract_pytorch_decoder_name,
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


def filter_stats(
    stats: list[sinter.TaskStats],
    *,
    p_list: list[float],
    max_iter: int,
) -> list[sinter.TaskStats]:
    """
    Filter the list of `sinter.TaskStats` to only include those consistent 
    with the given `p_list` and `max_iter`.
    """
    filtered: list[sinter.TaskStats] = []
    for s in stats:
        if s.json_metadata["p"] not in p_list:
            continue
        if "max_iter" in s.json_metadata and s.json_metadata["max_iter"] != max_iter:
            continue
        filtered.append(s)
    return filtered


def validate_stats(
    stats: list[sinter.TaskStats],
    *,
    p_list: list[float],
    max_shots: int,
    max_errors: int,
) -> bool:
    """
    Validate that: 
    - The list `stats` covers all p in `p_list` and no other p.
    - Each element of `stats` has enough shots to meet `max_shots` or enough errors to meet `max_errors`.

    Return `True` if validation succeeds, `False` otherwise.
    """
    covered_p: set[float] = set(s.json_metadata["p"] for s in stats)
    if covered_p != set(p_list):
        return False
    for s in stats:
        if s.shots < max_shots and s.errors < max_errors:
            return False
    return True


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
) -> list[sinter.TaskStats]:
    """
    Load and merge PyTorch decoders' stats and baseline decoders' stats into a single list.
    If any of the CSV files does not exist or only contains partial results, call `st.warning()` 
    and `st.stop()`.
    """
    all_stats: list[sinter.TaskStats] = []

    # Load PyTorch decoders' stats.
    for run_dir in run_dirs:
        csv_path = run_dir / BENCHMARK_CSV_FILENAME
        if not csv_path.exists():
            st.warning("Missing data. Run benchmark first.")
            st.stop()
        stats = sinter.read_stats_from_csv_files(csv_path)
        stats = filter_stats(stats, p_list=p_list, max_iter=max_iter)
        if not validate_stats(stats, p_list=p_list, max_shots=max_shots, max_errors=max_errors):
            st.warning("Missing data. Run benchmark first.")
            st.stop()
        all_stats.extend(stats)

    # Load baseline decoders' stats.
    for decoder in baseline_decoders:
        csv_path = get_baseline_csv_path(
            code, noise_model, d, rounds, basis, decoder
        )
        if not csv_path.exists():
            st.warning("Missing data. Run benchmark first.")
            st.stop()
        stats = sinter.read_stats_from_csv_files(csv_path)
        stats = filter_stats(stats, p_list=p_list, max_iter=max_iter)
        if not validate_stats(stats, p_list=p_list, max_shots=max_shots, max_errors=max_errors):
            st.warning("Missing data. Run benchmark first.")
            st.stop()
        all_stats.extend(stats)

    return all_stats


def main():
    st.set_page_config(page_title="Decoder Benchmark", layout="wide", page_icon="📈")

    # Prevent truncation of multiselect chip labels so decoder names are fully visible
    st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb=select] span {
                max-width: 500px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Monte Carlo Benchmark")

    # Discover all run_dirs
    run_dirs = discover_run_dirs()
    if len(run_dirs) == 0:
        st.warning("No trained PyTorch decoders found. Train a model first.")
        st.stop()
    if not is_unique(run_dirs):
        raise Exception("Duplicate run_dirs found.")

    # Sidebar: hierarchical selection
    with st.sidebar:
        st.subheader("Select QEC parameters")
        grouped = group_run_dirs_by_code_and_noise(run_dirs)  # Group run_dirs by (code, noise_model)
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
        run_dirs = grouped[selected_code_noise_pair]  # Filter run_dirs by selected (code, noise_model)

        grouped = group_run_dirs_by_d_rounds_basis(run_dirs)  # Group run_dirs by (d, rounds, basis)
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
        run_dirs = grouped[selected_d_rounds_basis_triple]  # Filter run_dirs by selected (d, rounds, basis)

        st.subheader("Select PyTorch decoder(s)")
        selected_run_dirs = st.multiselect(
            "PyTorch decoder(s) to benchmark",
            options=sorted(run_dirs),
            default=None,
            format_func=extract_pytorch_decoder_name,
        )

        st.subheader("Select baseline decoder(s)")
        selected_baseline_decoders = st.multiselect(
            "Baseline decoder(s) to benchmark against",
            options=BASELINE_DECODERS,
            default=None,
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
            help="Stops Monte Carlo sampling after taking this many shots."
        )
        max_errors = st.number_input(
            "Max number of failures",
            value=DEFAULT_MAX_ERRORS,
            min_value=1,
            help="Stops Monte Carlo sampling after having seen this many failures."
        )

        if st.button("Run benchmark", type="primary"):
            if len(selected_run_dirs) == 0 and len(selected_baseline_decoders) == 0:
                st.warning("Please select at least one PyTorch decoder or baseline decoder.")
                st.stop()

            with st.spinner("Running benchmark..."):
                try:
                    run_benchmark(
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
                    st.success("Benchmark complete.")
                except Exception as e:
                    st.error(str(e))
                    st.stop()

    # Main: plot
    if len(selected_run_dirs) == 0 and len(selected_baseline_decoders) == 0:
        st.warning("Please select at least one decoder in the sidebar.")
        st.stop()

    stats = load_and_merge_stats(
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

    st.subheader("Logical Error Rate (per shot) vs Physical Error Rate")
    fig1, ax1 = plt.subplots(1, 1)

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

    sinter.plot_error_rate(
        ax=ax1,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        filter_func=filter_func,
        group_func=group_func,
        plot_args_func=lambda index, group_key: {
            "linestyle": group_key["linestyle"],
        },
    )
    ax1.loglog()
    ax1.grid(axis='y')
    ax1.set_title(f"{code}, {noise_model}, d={d}, rounds={rounds}, basis={basis}")
    ax1.set_ylabel("Logical Error Rate (per shot)")
    ax1.set_xlabel("Physical Error Rate")
    ax1.legend()
    st.pyplot(fig1)

    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", dpi=150)
    st.download_button(
        "Download LER (per shot) vs PER plot as PNG",
        data=buf1.getvalue(),
        file_name="benchmark_LER_per_shot_vs_PER.png",
        mime="image/png",
    )

    # Add a visual line separator between the two plots
    st.markdown("---")

    st.subheader("Logical Error Rate (per round) vs Physical Error Rate")
    fig2, ax2 = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax2,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        filter_func=filter_func,
        group_func=group_func,
        failure_units_per_shot_func=lambda stat: stat.json_metadata["rounds"],
        plot_args_func=lambda index, group_key: {
            "linestyle": group_key["linestyle"],
        },
    )
    ax2.loglog()
    ax2.grid(axis='y')
    ax2.set_title(f"{code}, {noise_model}, d={d}, rounds={rounds}, basis={basis}")
    ax2.set_ylabel("Logical Error Rate (per round)")
    ax2.set_xlabel("Physical Error Rate")
    ax2.legend()
    st.pyplot(fig2)

    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", dpi=150)
    st.download_button(
        "Download LER (per round) vs PER plot as PNG",
        data=buf2.getvalue(),
        file_name="benchmark_LER_per_round_vs_PER.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
