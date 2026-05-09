"""Headless multi-decoder Monte Carlo benchmark entry point.

Hoisted from `pages/custom_benchmark_page.py:_run_all_benchmarks` so it can be
called from a plain Python script without launching Streamlit -- useful for
long-running jobs on remote servers (e.g. under tmux/nohup).

The Streamlit page calls this same function from its background thread, so the
UI and headless paths share one implementation.

Note: like the rest of `benchmark-app`, this module imports the top-level
`constants`, `experiment_factory`, and `torchdecoder_utils` modules. Callers
must run with `benchmark-app/` on `sys.path` (i.e. CWD = `benchmark-app/`).
"""

import threading
from pathlib import Path
from typing import Iterable, Optional

from qecdec.experiments import Experiment

from constants import BASELINES_CSV_DIR
from experiment_factory import create_experiment
from torchdecoder_utils import (
    extract_pytorch_decoder_name,
    get_ckpt_path,
    load_model_config_from_run_dir,
)

from ..params import BenchTaskParams, QECParams
from .baselines_runner import run_baseline_benchmark
from .collector_params import CollectorParams
from .stats_io import get_torchdecoder_csv_path
from .torchdecoder_runner import run_torchdecoder_benchmark


def run_custom_benchmark(
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    baseline_decoders: Iterable[str] = (),
    torchdecoder_run_dirs: Iterable[Path] = (),
    baseline_csv_dir: Path = BASELINES_CSV_DIR,
    experiments: Optional[dict[float, Experiment]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run a multi-decoder custom Monte Carlo benchmark synchronously.

    Iterates over the requested baseline decoders and PyTorch run directories,
    invoking the appropriate per-decoder runner for each. Results are appended
    to CSVs under `baseline_csv_dir` (baselines) or under each `run_dir`
    (PyTorch); existing CSVs are resumed in place.

    Parameters
    ----------
    qec_params, benchtask_params, collector_params
        QEC code spec, decoder configs + p_list, and Monte Carlo collector
        knobs (batch size, shot/error caps, parallelism).
    baseline_decoders
        Names of qecdec baseline decoders to run (e.g. "BP", "MWPM",
        "RelayBP"). Each must have a matching entry in
        `benchtask_params.baseline_decoder_params`.
    torchdecoder_run_dirs
        Lightning run directories whose checkpoints should be benchmarked.
        Each must have a model config and `last.ckpt` discoverable via
        `torchdecoder_utils`.
    baseline_csv_dir
        Root directory for baseline CSV outputs. Defaults to the project's
        `benchmark-app/baselines-results/`.
    experiments
        Optional pre-built `{p: Experiment}` map. If omitted, experiments are
        constructed via `experiment_factory.create_experiment` (auto-detecting
        whether the code requires loading from a stim file).
    stop_event
        Optional `threading.Event` for cooperative cancellation. Checked
        between decoders and inside `collect_stats` between batches.
    """
    if experiments is None:
        load_circuit_from_file = qec_params.code.startswith("BB_") or (
            qec_params.code == "HexColorCode" and qec_params.noise_model == "Superdense"
        )
        experiments = {
            p: create_experiment(
                qec_params, p, load_circuit_from_file=load_circuit_from_file
            )
            for p in benchtask_params.p_list
        }

    for decoder_name in baseline_decoders:
        if stop_event is not None and stop_event.is_set():
            return
        run_baseline_benchmark(
            baseline_csv_dir,
            decoder_name,
            qec_params=qec_params,
            benchtask_params=benchtask_params,
            collector_params=collector_params,
            stop_event=stop_event,
            experiments=experiments,
        )

    for run_dir in torchdecoder_run_dirs:
        if stop_event is not None and stop_event.is_set():
            return
        run_torchdecoder_benchmark(
            csv_path=get_torchdecoder_csv_path(run_dir),
            decoder_name=extract_pytorch_decoder_name(run_dir),
            model_cfg=load_model_config_from_run_dir(run_dir),
            ckpt_path=get_ckpt_path(run_dir),
            qec_params=qec_params,
            benchtask_params=benchtask_params,
            collector_params=collector_params,
            stop_event=stop_event,
            experiments=experiments,
        )
