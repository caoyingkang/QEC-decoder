"""
Streamlit app for Monte Carlo benchmarking of trained decoder models.

Select runs, run or load cached benchmarks, and plot logical error rate vs
physical error rate.
"""
import os
import subprocess
import sys
from pathlib import Path

import sinter
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_PYTORCH_ROOT = _APP_DIR.parent
_RUNS_ROOT = _PYTORCH_ROOT / "runs"
_BASELINES_ROOT = _PYTORCH_ROOT / "baselines"
_SCRIPTS_DIR = _PYTORCH_ROOT / "scripts"
BENCHMARK_RESULTS_FILENAME = "benchmark_results.csv"

DEFAULT_P_LIST = [0.004, 0.006, 0.008, 0.01, 0.012]
DEFAULT_MAX_SHOTS = 10_000_000
DEFAULT_MAX_ERRORS = 100


def get_baselines_path(d: int, rounds: int, basis: str) -> Path:
    code_dir = f"d={d}_rounds={rounds}_basis={basis}"
    return _BASELINES_ROOT / "rotated_surface_code_memory" / code_dir / BENCHMARK_RESULTS_FILENAME


def get_code_dir_from_run(run_dir: Path) -> tuple[int, int, str]:
    for p in run_dir.parts:
        if p.startswith("d=") and "_rounds=" in p and "_basis=" in p:
            d = int(p.split("d=")[1].split("_")[0])
            rounds = int(p.split("rounds=")[1].split("_")[0])
            basis = p.split("basis=")[1]
            return d, rounds, basis
    raise ValueError(f"Cannot parse code config from run dir: {run_dir}")


def discover_runs() -> list[tuple[Path, str]]:
    """Return list of (run_dir, display_label)."""
    runs = []
    for p in _RUNS_ROOT.rglob("checkpoints/best_model.ckpt"):
        run_dir = p.parent.parent
        if run_dir.name.startswith("run_"):
            try:
                d, rounds, basis = get_code_dir_from_run(run_dir)
                model_name = run_dir.parent.name
                label = f"d={d} rounds={rounds} / {model_name} / {run_dir.name}"
                runs.append((run_dir, label))
            except ValueError:
                pass
    return sorted(runs, key=lambda x: x[1])


def load_and_merge_stats(selected_runs: list[Path]) -> list:
    """Load baseline + per-run stats and merge into single list for plotting."""
    all_stats = []
    seen_configs = set()

    for run_dir in selected_runs:
        d, rounds, basis = get_code_dir_from_run(run_dir)
        config_key = (d, rounds, basis)

        # Load baselines once per config
        if config_key not in seen_configs:
            baselines_path = get_baselines_path(d, rounds, basis)
            if baselines_path.exists():
                stats = sinter.read_stats_from_csv_files(baselines_path)
                all_stats.extend(stats)
            seen_configs.add(config_key)

        # Load per-run learned DMemBP results
        run_results_path = run_dir / BENCHMARK_RESULTS_FILENAME
        if run_results_path.exists():
            stats = sinter.read_stats_from_csv_files(run_results_path)
            all_stats.extend(stats)

    return all_stats


def run_benchmark(run_dirs: list[Path], p_list: list[float], max_shots: int, max_errors: int) -> None:
    """Run benchmark_decoder.py as subprocess."""
    run_paths = [str(r) for r in run_dirs]
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "benchmark_decoder.py"),
        *run_paths,
        "--p-list", *[str(p) for p in p_list],
        "--max-shots", str(max_shots),
        "--max-errors", str(max_errors),
    ]
    project_root = _PYTORCH_ROOT.parent
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark script exited with code {result.returncode}")


def main():
    st.set_page_config(page_title="Decoder Benchmark", layout="wide")
    st.title("Monte Carlo Decoder Benchmark")

    runs_with_labels = discover_runs()
    if not runs_with_labels:
        st.warning("No runs found. Train a model first, then runs will appear here.")
        return

    # Sidebar: run selection and benchmark params
    with st.sidebar:
        st.subheader("Select runs")
        run_options = {label: run_dir for run_dir, label in runs_with_labels}
        selected_labels = st.multiselect(
            "Runs to plot",
            options=list(run_options.keys()),
            default=[],
        )
        selected_runs = [run_options[l] for l in selected_labels]

        st.subheader("Benchmark parameters")
        p_list_str = st.text_input(
            "p_list (comma-separated)",
            value=", ".join(str(p) for p in DEFAULT_P_LIST),
        )
        try:
            p_list = [float(x.strip()) for x in p_list_str.split(",") if x.strip()]
        except ValueError:
            p_list = DEFAULT_P_LIST
        max_shots = st.number_input("max_shots", value=DEFAULT_MAX_SHOTS, min_value=1000)
        max_errors = st.number_input("max_errors", value=DEFAULT_MAX_ERRORS, min_value=1)
        num_workers = max(1, (os.cpu_count() or 1) - 1)

        if st.button("Run benchmark", type="primary") and selected_runs:
            with st.spinner("Running benchmark..."):
                try:
                    run_benchmark(selected_runs, p_list, max_shots, max_errors)
                    st.success("Benchmark complete.")
                except Exception as e:
                    st.error(str(e))

    # Main: plot
    if not selected_runs:
        st.info("Select one or more runs from the sidebar to plot.")
        return

    stats = load_and_merge_stats(selected_runs)
    if not stats:
        st.warning(
            "No benchmark data found. Run the benchmark first, or ensure "
            "baselines exist for the selected code configs."
        )
        return

    st.subheader("Logical error rate vs physical error rate")
    fig, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        group_func=lambda stat: stat.json_metadata.get("decoder", "unknown"),
        x_func=lambda stat: stat.json_metadata.get("p", 0),
        plot_args_func=lambda index, group_key: {
            "linestyle": "solid" if "bp" in str(group_key) or "dmembp" in str(group_key) else "dashed",
        },
    )
    ax.loglog()
    ax.grid()
    ax.set_ylabel("Logical Error Probability (per shot)")
    ax.set_xlabel("Physical Error Rate")
    ax.legend()
    st.pyplot(fig)

    # Export as PNG
    buf = __import__("io", fromlist=["BytesIO"]).BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    st.download_button(
        "Download plot as PNG",
        data=buf.getvalue(),
        file_name="benchmark_plot.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
