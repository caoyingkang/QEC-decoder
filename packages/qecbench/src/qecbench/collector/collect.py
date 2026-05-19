"""Monte Carlo collection loop — replacement for sinter.collect."""

from pathlib import Path
import threading
import time
from typing import Optional

import humanize
from rich import print as rprint
import stim

from ..decoder_wrapper import BenchmarkDecoder
from ..stats import BenchmarkStats, TaskMetadata
from .collect_parallel import _run_parallel_collect
from .collect_serial import _run_serial_collect
from .utils import _info


def _collect_stats(
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    taskmetadata: TaskMetadata,
    *,
    batch_size: int,
    shots_cap: int,
    errors_cap: int,
    num_parallel_workers: int,
    poll_interval_sec: float = 1.0,
    csv_path: Optional[Path | str] = None,
    verbose: bool = True,
    th_stop_event: Optional[threading.Event] = None,
) -> BenchmarkStats:
    """Collect Monte Carlo benchmark statistics for a single task.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to sample from.
    decoder : Decoder
        Decoder to evaluate.
    taskmetadata : TaskMetadata
        Metadata identifying this benchmark task.
    batch_size : int
        Number of shots in one batch.
    shots_cap, errors_cap : int
        The benchmark is considered complete when either the total number of shots
        reaches ``shots_cap``, or the number of shots with incorrect observable
        predictions reaches ``errors_cap``.
    num_parallel_workers : int
        Number of parallel worker processes (0 = serial, >0 = multiprocessing).
    poll_interval_sec : float
        In multiprocessing mode, how often in seconds the main process polls the workers
        to check global completion of the benchmark task. Ignored in serial mode.
    csv_path : Path or str or None
        If specified, resume from and save results to this CSV file. If the file does not
        exist, it (and its parent directories) will be created on save.
    verbose : bool
        Whether to print progress to stdout.
    th_stop_event : threading.Event or None
        If provided, the collection loop checks this event every batch (serial mode) or
        every poll interval (parallel mode) and exits early when it is set.

    Returns
    -------
    BenchmarkStats
    """
    # --- Resume check ---------------------------------------------------------
    csv_path = Path(csv_path) if csv_path is not None else None
    if csv_path is not None:
        csv_stats_list = BenchmarkStats.load_csv(csv_path)
        prev_stats = BenchmarkStats.find_by_metadata(csv_stats_list, taskmetadata)
    else:
        prev_stats = None

    if prev_stats is not None:
        if prev_stats.is_complete(shots_cap, errors_cap):
            if verbose:
                _info("Task already complete.")
                rprint(taskmetadata)
            return prev_stats
        else:
            resuming = True
            shots_cap = shots_cap - prev_stats.shots
            errors_cap = errors_cap - prev_stats.obser_errors
            if verbose:
                _info("Resume from incomplete task.")
                rprint(taskmetadata)
    else:
        resuming = False
        if verbose:
            _info("Start new task.")
            rprint(taskmetadata)

    # --- Run MC loop ----------------------------------------------------------
    if verbose:
        t_start = time.time()

    if num_parallel_workers <= 0:  # serial mode
        stats = _run_serial_collect(
            dem,
            decoder,
            taskmetadata,
            batch_size=batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
            verbose=verbose,
            th_stop_event=th_stop_event,
        )
    else:  # multiprocessing mode
        stats = _run_parallel_collect(
            dem,
            decoder,
            taskmetadata,
            batch_size=batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
            num_workers=num_parallel_workers,
            poll_interval_sec=poll_interval_sec,
            verbose=verbose,
            th_stop_event=th_stop_event,
        )

    if verbose:
        elapsed = time.time() - t_start
        elapsed_str = humanize.precisedelta(elapsed)
        speed_str = f" ({stats.shots / elapsed:.1f} shots/s)" if elapsed > 1.0 else ""
        if th_stop_event is not None and th_stop_event.is_set():
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
        BenchmarkStats.upsert(csv_stats_list, stats)
        BenchmarkStats.save_csv(csv_stats_list, csv_path)

    return stats
