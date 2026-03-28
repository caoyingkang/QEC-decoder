"""Sinter-based Monte Carlo benchmark page."""

import os
from io import BytesIO

import matplotlib.pyplot as plt
import sinter
import streamlit as st

from bench.constants import BASELINE_DECODERS
from bench.sinter_bench.collector_params import CollectorParams
from bench.sinter_bench.stats_io import (
    get_torchdecoder_csv_path,
    load_and_merge_stats,
)
from bench.sinter_bench.baselines_runner import (
    run_MWPM_benchmark,
    run_BP_benchmark,
)
from bench.sinter_bench.torchdecoder_runner import run_torchdecoder_benchmark
from torchdecoder_utils import (
    extract_pytorch_decoder_name,
    load_model_config_from_run_dir,
    get_ckpt_path,
)
from constants import BASELINES_CSV_DIR
from shared_ui import (
    render_sidebar_baselines_selection,
    render_sidebar_bench_task_selection,
    render_sidebar_collector_selection_common,
    render_qec_selection,
    render_decoder_selection,
    stop_if_no_decoders_selected,
    render_missing_data_warning_and_benchmark_button,
)


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


selected_baseline_decoders = render_sidebar_baselines_selection()
benchtask_params = render_sidebar_bench_task_selection()
collector_params = _render_sidebar_collector_selection()
qec_params, run_dirs = render_qec_selection()
selected_run_dirs = render_decoder_selection(run_dirs)
stop_if_no_decoders_selected(selected_run_dirs, selected_baseline_decoders)

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
            for run_dir in pending_run_dirs:
                run_torchdecoder_benchmark(
                    csv_path=get_torchdecoder_csv_path(run_dir),
                    decoder_name=extract_pytorch_decoder_name(run_dir),
                    model_cfg=load_model_config_from_run_dir(run_dir),
                    ckpt_path=get_ckpt_path(run_dir),
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
                )
            if "MWPM" in pending_baseline_decoders:
                run_MWPM_benchmark(
                    BASELINES_CSV_DIR,
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
                )
            if "BP" in pending_baseline_decoders:
                run_BP_benchmark(
                    BASELINES_CSV_DIR,
                    qec_params=qec_params,
                    benchtask_params=benchtask_params,
                    collector_params=collector_params,
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


def _filter_func(stat: sinter.TaskStats) -> bool:
    cond1 = stat.json_metadata["p"] in benchtask_params.p_list
    if "max_iter" in stat.json_metadata:
        cond2 = stat.json_metadata["max_iter"] == benchtask_params.max_iter
    else:
        cond2 = True
    return cond1 and cond2


def _group_func(stat: sinter.TaskStats) -> dict:
    decoder = stat.json_metadata["decoder"]
    return {
        "label": decoder,
        "linestyle": "dashed" if decoder in BASELINE_DECODERS else "solid",
    }


fig, ax = plt.subplots(1, 1)
common_kwargs = dict(
    ax=ax,
    stats=stats,
    x_func=lambda stat: stat.json_metadata["p"],
    filter_func=_filter_func,
    group_func=_group_func,
    plot_args_func=lambda index, group_key: {"linestyle": group_key["linestyle"]},
)
if ler_mode == "per shot":
    sinter.plot_error_rate(**common_kwargs)
    ax.set_ylabel("LER per shot")
else:
    sinter.plot_error_rate(
        **common_kwargs,
        failure_units_per_shot_func=lambda stat: stat.json_metadata["rounds"],
    )
    ax.set_ylabel("LER per round")

ax.loglog()
ax.grid(axis="y")
ax.set_title(
    f"{qec_params.code}, {qec_params.noise_model}, d={qec_params.d}, rounds={qec_params.rounds}, basis={qec_params.basis}"
)
ax.set_xlabel("PER")
ax.legend()

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
