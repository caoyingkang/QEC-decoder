"""
Streamlit app for Monte Carlo benchmarking of QEC decoders.
Run with: `uv run streamlit run benchmark-app/app.py` (from repo root)
"""

from pathlib import Path
from qecbench import CollectorParams, TaskMetadata, run_benchmark
import qecdec
import queue
import streamlit as st
import threading

from constants import CIRCUITS_ROOT
from ui import render_sidebar_collector_selection, render_task_selection
from utils import get_csv_path

qecdec.circuits.BB_144_12_12_Circuit.load_dir = CIRCUITS_ROOT / "BB_144_12_12_Circuit"


def thread_target(
    task_metadata: TaskMetadata,
    collector_params: CollectorParams,
    csv_path: Path,
    stop_event: threading.Event,
    exc_queue: queue.Queue,
) -> None:
    """Run benchmark task. Exit early if `stop_event` is set.
    Put any exception that occurs into `exc_queue`.
    """
    try:
        run_benchmark(
            task_metadata,
            collector_params,
            csv_path=csv_path,
            stop_event=stop_event,
        )
    except Exception as e:
        exc_queue.put(e)


@st.fragment(run_every=2)
def monitor(
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
def benchmark_running_modal(
    task_metadata: TaskMetadata, collector_params: CollectorParams, csv_path: Path
) -> None:
    """Modal dialog shown while a benchmark is in progress.

    Blocks all interaction with the rest of the app.  The only way to dismiss
    the dialog is by clicking the "Stop benchmark" button or waiting for the
    benchmark to finish.
    """
    stop_event = threading.Event()
    exc_queue = queue.Queue()
    thread = threading.Thread(
        target=thread_target,
        args=(
            task_metadata,
            collector_params,
            csv_path,
            stop_event,
            exc_queue,
        ),
        daemon=True,
    )
    thread.start()
    monitor(thread, stop_event, exc_queue)


if __name__ == "__main__":
    st.set_page_config(page_title="Decoder Benchmark", layout="wide", page_icon="📈")
    st.title("Monte Carlo Benchmark")

    collector_params = render_sidebar_collector_selection()
    task_metadata = render_task_selection()
    csv_path = get_csv_path(
        task_metadata.circuit_name,
        task_metadata.circuit_params,
        task_metadata.decoder_name,
    )

    # from rich import print as rprint

    # rprint("collector_params:\n", collector_params)
    # rprint("task_metadata:\n", task_metadata)
    # rprint("csv_path:\n", csv_path)

    if st.button("Run benchmark", type="primary"):
        benchmark_running_modal(task_metadata, collector_params, csv_path)
    st.stop()
