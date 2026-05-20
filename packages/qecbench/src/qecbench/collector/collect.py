"""Monte Carlo collection loop — replacement for sinter.collect."""

import threading
import time
from typing import Optional

import humanize
from rich import print as rprint
import stim

from ..decoder_wrapper import BenchmarkDecoder
from ..task import TaskMetadata, TaskStats
from .collect_parallel import _run_parallel_collect
from .collect_serial import _run_serial_collect
from .params import CollectorParams
from .utils import _info


def _collect_stats(
    *,
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    taskmetadata: TaskMetadata,
    collector_params: CollectorParams,
    verbose: bool,
    th_stop_event: Optional[threading.Event] = None,
) -> TaskStats:
    """Collect Monte Carlo benchmark statistics for a single task.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to sample from.
    decoder : BenchmarkDecoder
        Decoder to evaluate.
    taskmetadata : TaskMetadata
        Metadata identifying this benchmark task.
    collector_params: CollectorParams
        Monte Carlo collector parameters
    verbose : bool
        Whether to print progress to stdout.
    th_stop_event : threading.Event or None
        If provided, the collection loop checks this event every batch (serial
        mode) or every poll interval (parallel mode) and exits early when it
        is set.

    Returns
    -------
    TaskStats
    """
    csv_path = collector_params.csv_path
    shots_cap = collector_params.shots_cap
    errors_cap = collector_params.errors_cap

    # --- Resume check ---------------------------------------------------------
    if csv_path is not None:
        csv_stats_list = TaskStats.load_csv(csv_path)
        prev_stats = TaskStats.find_by_metadata(csv_stats_list, taskmetadata)
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
            shots_cap -= prev_stats.shots
            errors_cap -= prev_stats.obser_errors
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

    if collector_params.use_multiprocessing:  # multiprocessing mode
        stats = _run_parallel_collect(
            dem=dem,
            decoder=decoder,
            metadata=taskmetadata,
            batch_size=collector_params.batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
            num_workers=collector_params.num_parallel_workers,
            poll_interval_sec=1.0,
            verbose=verbose,
            th_stop_event=th_stop_event,
        )
    else:  # serial mode
        stats = _run_serial_collect(
            dem=dem,
            decoder=decoder,
            metadata=taskmetadata,
            batch_size=collector_params.batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
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
        TaskStats.upsert(csv_stats_list, stats)
        TaskStats.save_csv(csv_stats_list, csv_path)

    return stats
