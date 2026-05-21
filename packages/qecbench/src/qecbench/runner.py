from pathlib import Path
import threading
import time
from typing import Optional

import humanize
from qecdec.circuits import create_circuit_with_uniform_error_rate
from qecdec.decoders import create_decoder
from rich import print as rprint

from .collector import CollectorParams, _collect_stats_parallel, _collect_stats_serial
from .decoder_wrapper import BenchmarkDecoder
from .task import TaskMetadata, TaskStats
from .utils import _info


def run_benchmark(
    task_metadata: TaskMetadata,
    collector_params: CollectorParams,
    *,
    verbose: bool = True,
    csv_path: Optional[str | Path] = None,
    stop_event: Optional[threading.Event] = None,
) -> TaskStats:
    """Run Monte Carlo benchmark.

    Parameters
    ----------
    task_metadata: TaskMetadata
        Metadata of the benchmark task (circuit & decoder configs).
    collector_params: CollectorParams
        Parameters of statistics collector.
    verbose : bool
        Whether to print progress to stdout.
    csv_path : str or Path or None
        If specified, resume from and save results to this CSV file. If the
        file does not exist, it (and its parent directories) will be created
        on save.
    stop_event : threading.Event or None
        If provided, the collection loop checks this event every batch (serial
        mode) or every poll interval (parallel mode) and exits early when it
        is set.
    """
    csv_path = Path(csv_path) if csv_path is not None else None
    shots_cap = collector_params.shots_cap
    errors_cap = collector_params.errors_cap

    # --- Resume check ---------------------------------------------------------
    if csv_path is not None:
        csv_stats_list = TaskStats.load_csv(csv_path)
        prev_stats = TaskStats.find_by_metadata(csv_stats_list, task_metadata)
    else:
        prev_stats = None

    if prev_stats is not None:
        if prev_stats.is_complete(shots_cap, errors_cap):
            if verbose:
                _info("Task already complete.")
                rprint(task_metadata)
            return prev_stats
        else:
            resuming = True
            shots_cap -= prev_stats.shots
            errors_cap -= prev_stats.obser_errors
            if verbose:
                _info("Resume from incomplete task.")
                rprint(task_metadata)
    else:
        resuming = False
        if verbose:
            _info("Start new task.")
            rprint(task_metadata)

    # --- Build circuit and decoder ----------------------------------------------
    circuit = create_circuit_with_uniform_error_rate(
        task_metadata.circuit_name,
        task_metadata.error_rate,
        **task_metadata.circuit_params,
    )
    decoder = create_decoder(
        task_metadata.decoder_name,
        circuit.chkmat,
        circuit.prior,
        **task_metadata.decoder_params,
    )
    benchmark_decoder = BenchmarkDecoder(decoder, circuit.obsmat)

    # --- Run MC loop ----------------------------------------------------------
    if verbose:
        t_start = time.time()

    if collector_params.use_multiprocessing:  # multiprocessing mode
        stats = _collect_stats_parallel(
            dem=circuit.stim_dem,
            decoder=benchmark_decoder,
            metadata=task_metadata,
            batch_size=collector_params.batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
            num_workers=collector_params.num_parallel_workers,
            poll_interval_sec=1.0,
            verbose=verbose,
            th_stop_event=stop_event,
        )
    else:  # serial mode
        stats = _collect_stats_serial(
            dem=circuit.stim_dem,
            decoder=benchmark_decoder,
            metadata=task_metadata,
            batch_size=collector_params.batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
            verbose=verbose,
            th_stop_event=stop_event,
        )

    if verbose:
        elapsed = time.time() - t_start
        elapsed_str = humanize.precisedelta(elapsed)
        speed_str = f" ({stats.shots / elapsed:.1f} shots/s)" if elapsed > 1.0 else ""
        if stop_event is not None and stop_event.is_set():
            _info(
                f"Task stopped. Collected {stats.shots:,} shots in {elapsed_str}{speed_str}."
            )
        else:
            _info(
                f"Task completed. Collected {stats.shots:,} shots in {elapsed_str}{speed_str}."
            )
        print()

    # Merge with previous partial stats if resuming
    if resuming:
        stats.merge(prev_stats)

    # --- Save -----------------------------------------------------------------
    if csv_path is not None:
        TaskStats.upsert(csv_stats_list, stats)
        TaskStats.save_csv(csv_stats_list, csv_path)

    return stats
