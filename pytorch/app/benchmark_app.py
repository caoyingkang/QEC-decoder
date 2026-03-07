"""
Streamlit app for Monte Carlo benchmarking of trained decoder models.

Select runs, run or load cached benchmarks, and plot logical error rate vs physical error rate.
"""
import subprocess
import sys
from io import BytesIO
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import sinter
import streamlit as st

from utils import is_unique

_PYTORCH_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _PYTORCH_ROOT / "app"
_RUNS_ROOT = _PYTORCH_ROOT / "runs"
_BASELINES_ROOT = _PYTORCH_ROOT / "baselines"
_SCRIPTS_DIR = _PYTORCH_ROOT / "scripts"
BENCHMARK_RESULTS_FILENAME = "benchmark_results.csv"

BASELINE_DECODERS = ["pymatching", "bp"]  # Used for plot styling (dashed lines)
BASELINE_OPTIONS = {"MWPM": "pymatching", "BP": "bp"}

DEFAULT_P_LIST = [0.004, 0.006, 0.008, 0.01, 0.012]
DEFAULT_MAX_SHOTS = 1_000_000
DEFAULT_MAX_ERRORS = 100


def get_baselines_path(d: int, rounds: int, basis: str) -> Path:
    code_dir = f"d={d}_rounds={rounds}_basis={basis}"
    return _BASELINES_ROOT / "rotated_surface_code_memory" / code_dir / BENCHMARK_RESULTS_FILENAME


def get_config_from_run_dir(run_dir: Path) -> DictConfig:
    """Load config.yaml from a run directory."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def discover_runs() -> list[tuple[Path, str]]:
    """Return list of (run_dir, label)."""
    runs = []
    for p in _RUNS_ROOT.rglob("checkpoints/best_model.ckpt"):
        run_dir = p.parent.parent
        cfg = get_config_from_run_dir(run_dir)
        d = cfg.qec.d
        rounds = cfg.qec.rounds
        basis = cfg.qec.basis
        model_name = cfg.model.name
        label = f"d={d} rounds={rounds} basis={basis} / {model_name} / {run_dir.name}"
        runs.append((run_dir, label))
    return sorted(runs, key=lambda x: x[1])


def load_and_merge_stats(run_dirs: list[Path], baselines: list[str]) -> list[sinter.TaskStats]:
    """Load baseline + per-run stats and merge into single list for plotting."""
    # TODO: collect all CSV files first, then call sinter.read_stats_from_csv_files once
    all_stats = []
    seen_qec_settings = set()

    for run_dir in run_dirs:
        cfg = get_config_from_run_dir(run_dir)
        d = cfg.qec.d
        rounds = cfg.qec.rounds
        basis = cfg.qec.basis
        qec_setting = (d, rounds, basis)

        # Load baselines once per QEC setting
        if qec_setting not in seen_qec_settings:
            seen_qec_settings.add(qec_setting)
            baselines_path = get_baselines_path(d, rounds, basis)
            if not baselines_path.exists():
                pass  # TODO: handle this case
            stats = sinter.read_stats_from_csv_files(baselines_path)
            # Filter to only include selected baseline decoders
            # TODO: when benchmarking, save results with different decoder names to avoid this filter
            for s in stats:
                if s.json_metadata["decoder"] in baselines:
                    all_stats.append(s)

        # Load per-run learned DMemBP results
        run_results_path = run_dir / BENCHMARK_RESULTS_FILENAME
        if not run_results_path.exists():
            pass  # TODO: handle this case
        stats = sinter.read_stats_from_csv_files(run_results_path)
        all_stats.extend(stats)

    return all_stats


def run_benchmark(
    run_dirs: list[Path],
    p_list: list[float],
    max_shots: int,
    max_errors: int,
    baselines: list[str],
):
    """Run benchmark_decoder.py as subprocess."""
    run_dirs_as_strs = [str(r) for r in run_dirs]
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "benchmark_decoder.py"),
        *run_dirs_as_strs,
        "--p-list", *[str(p) for p in p_list],
        "--max-shots", str(max_shots),
        "--max-errors", str(max_errors),
        "--baselines", *baselines,
    ]
    project_root = _PYTORCH_ROOT.parent
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark script exited with code {result.returncode}")


def main():
    st.set_page_config(page_title="Decoder Benchmark", layout="wide", page_icon="📈")
    st.title("Monte Carlo Decoder Benchmark")

    rundirs_with_labels = discover_runs()
    if len(rundirs_with_labels) == 0:
        st.warning("No runs found. Train a model first, then runs will appear here.")
        return

    # Make sure labels are unique
    if not is_unique(label for _, label in rundirs_with_labels):
        raise Exception("Duplicate labels found.")

    # Sidebar: run selection and benchmark params
    with st.sidebar:
        st.subheader("Select runs")
        label2rundir = {l: r for r, l in rundirs_with_labels}
        selected_labels = st.multiselect(
            "Runs to plot",
            options=list(label2rundir.keys()),
            default=[],
        )
        selected_rundirs = [label2rundir[l] for l in selected_labels]

        st.subheader("Baseline decoders")
        selected_baseline_labels = st.multiselect(
            "Baseline decoder(s) to benchmark against",
            options=list(BASELINE_OPTIONS.keys()),
            default=list(BASELINE_OPTIONS.keys()),
        )
        selected_baselines = [BASELINE_OPTIONS[l] for l in selected_baseline_labels]

        st.subheader("Benchmark parameters")
        p_list_str = st.text_input(
            "p_list (comma-separated)",
            value=", ".join(str(p) for p in DEFAULT_P_LIST),
            help="List of physical error rates to benchmark at, separated by commas."
        )
        try:
            p_list = [float(x.strip()) for x in p_list_str.split(",") if x.strip()]
        except ValueError:
            st.error("Cannot parse p_list into a list of floats.")
            st.stop()
        if len(p_list) == 0:
            st.warning("p_list is empty.")
            st.stop()
        max_shots = st.number_input(
            "max_shots",
            value=DEFAULT_MAX_SHOTS,
            min_value=10_000,
            step=10_000,
            help="Stops Monte Carlo sampling after taking this many shots."
        )
        max_errors = st.number_input(
            "max_errors",
            value=DEFAULT_MAX_ERRORS,
            min_value=10,
            step=10,
            help="Stops Monte Carlo sampling after having seen this many decoding errors."
        )

        if st.button("Run benchmark", type="primary"):
            if len(selected_rundirs) == 0:
                st.warning("No runs selected.")
                st.stop()

            with st.spinner("Running benchmark..."):
                try:
                    run_benchmark(
                        selected_rundirs,
                        p_list,
                        max_shots,
                        max_errors,
                        selected_baselines,
                    )
                    st.success("Benchmark complete.")
                except Exception as e:
                    st.error(str(e))

    # Main: plot
    if len(selected_rundirs) == 0:
        st.info("Select one or more runs from the sidebar to plot.")
        st.stop()

    stats = load_and_merge_stats(selected_rundirs, selected_baselines)
    if len(stats) == 0:
        # TODO: ensure every selected run has benchmark data
        st.info("No benchmark data found. Run the benchmark first.")
        st.stop()

    st.subheader("Logical Error Rate vs Physical Error Rate")
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        # TODO: group by runs. How to do this gracefully?
        group_func=lambda stat: dict(
            label=", ".join([
                f"d={stat.json_metadata['d']}",
                f"rounds={stat.json_metadata['rounds']}",
                f"basis={stat.json_metadata['basis']}",
                f"decoder={stat.json_metadata['decoder']}",
            ]),
            linestyle="dashed" if stat.json_metadata["decoder"] in BASELINE_DECODERS else "solid",
        ),
        plot_args_func=lambda index, group_key: {
            "linestyle": group_key["linestyle"],
        },
    )
    ax.loglog()
    ax.grid(axis='y')
    ax.set_ylabel("Logical Error Rate (per shot)")
    ax.set_xlabel("Physical Error Rate")
    ax.legend()
    st.pyplot(fig)

    # TODO: show legend outside of the axis
    # TODO: add a new figure to show logical error rate per round

    # Export as PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    st.download_button(
        "Download plot as PNG",
        data=buf.getvalue(),
        file_name="benchmark_plot.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
