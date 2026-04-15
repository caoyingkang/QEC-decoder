"""Monte Carlo collection loop — replacement for sinter.collect."""

import multiprocessing
import multiprocessing.synchronize
from queue import Empty
import threading
import sys
import time
from pathlib import Path
from typing import Optional

import humanize
import numpy as np
import stim
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

from .decoder import BenchmarkDecoder
from .stats import BenchmarkStats, TaskMetadata
from ..types import Bit2DArray


def _info(message: str) -> None:
    rprint(f"[cyan]Info: {message}[/cyan]")


def _sample(
    sampler: stim.CompiledDemSampler,
    batch_size: int,
) -> tuple[Bit2DArray, Bit2DArray]:
    """Sample syndromes and observables from a `stim.CompiledDemSampler`.

    Returns
    -------
    syndromes : Bit2DArray
        Syndrome bits, shape=(batch_size, num_checks).
    observables : Bit2DArray
        Observable bits, shape=(batch_size, num_obsers).
    """
    syndromes, observables, _ = sampler.sample(shots=batch_size)
    return syndromes.astype(np.uint8), observables.astype(np.uint8)


def _run_serial_collect(
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    metadata: TaskMetadata,
    *,
    batch_size: int,
    shots_cap: int,
    errors_cap: int,
    verbose: bool,
    th_stop_event: Optional[threading.Event] = None,
) -> BenchmarkStats:
    """Single-process MC collection loop."""
    sampler = dem.compile_sampler()
    stats = BenchmarkStats(metadata=metadata)

    with Progress(
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=Console(file=sys.stdout),
        refresh_per_second=2,
        disable=not verbose,
    ) as progress:
        shots_task = progress.add_task("[cyan]Shots", total=shots_cap)
        errors_task = progress.add_task("[cyan]Errors", total=errors_cap)
        while not stats.is_complete(shots_cap, errors_cap):
            if th_stop_event is not None and th_stop_event.is_set():
                break
            syndromes, observables = _sample(sampler, batch_size)
            result = decoder.decode(syndromes)
            obser_correct_mask = np.all(result.obser_pred == observables, axis=1)
            stats.update(
                obser_correct_mask,
                result.synd_match_mask,
                result.decoding_iters,
            )
            progress.update(
                shots_task,
                completed=stats.shots,
            )
            progress.update(
                errors_task,
                completed=stats.obser_errors,
            )

    return stats


def _read_global_totals(
    num_workers: int,
    shots_arr,
    errors_arr,
    locks,
) -> tuple[int, int]:
    """
    Read total shots and observable errors collected by all workers.
    Executed in the main process only.
    """
    total_shots = 0
    total_errors = 0
    for i in range(num_workers):
        with locks[i]:
            total_shots += int(shots_arr[i])
            total_errors += int(errors_arr[i])
    return total_shots, total_errors


def _worker_loop(
    worker_id: int,
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    metadata: TaskMetadata,
    batch_size: int,
    seed: int,
    shots_arr,
    errors_arr,
    locks: list[multiprocessing.synchronize.Lock],
    mp_stop_event: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker loop: publish cumulative stats to shared arrays `shots_arr` and
    `errors_arr`; exit when `mp_stop_event` is set."""
    stats = BenchmarkStats(metadata=metadata)
    try:
        sampler = dem.compile_sampler(seed=seed)
        while not mp_stop_event.is_set():
            syndromes, observables = _sample(sampler, batch_size)
            result = decoder.decode(syndromes)
            obser_correct_mask = np.all(result.obser_pred == observables, axis=1)
            stats.update(
                obser_correct_mask,
                result.synd_match_mask,
                result.decoding_iters,
            )
            with locks[worker_id]:
                shots_arr[worker_id] = stats.shots
                errors_arr[worker_id] = stats.obser_errors
    finally:  # Always publish the final stats so the main process will not hang
        result_queue.put(stats)


def _run_parallel_collect(
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    metadata: TaskMetadata,
    *,
    batch_size: int,
    shots_cap: int,
    errors_cap: int,
    num_workers: int,
    poll_interval_sec: float,
    verbose: bool,
    th_stop_event: Optional[threading.Event] = None,
) -> BenchmarkStats:
    """Multi-processing MC collection loop."""
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, but got {num_workers}.")
    if poll_interval_sec <= 0:
        raise ValueError(
            f"poll_interval_sec must be positive, but got {poll_interval_sec}."
        )

    # --- Create shared memory arrays -------------------------------------
    shots_arr = multiprocessing.Array("q", num_workers, lock=False)
    errors_arr = multiprocessing.Array("q", num_workers, lock=False)
    locks = [multiprocessing.Lock() for _ in range(num_workers)]
    mp_stop_event = multiprocessing.Event()
    result_queue = multiprocessing.Queue()

    # --- Create worker processes -----------------------------------------
    rng = np.random.default_rng()
    seeds = rng.integers(2**31, size=num_workers)
    processes: list[multiprocessing.Process] = []
    for i in range(num_workers):
        args = (
            i,
            dem,
            decoder,
            metadata,
            batch_size,
            int(seeds[i]),
            shots_arr,
            errors_arr,
            locks,
            mp_stop_event,
            result_queue,
        )
        processes.append(multiprocessing.Process(target=_worker_loop, args=args))

    # --- Start worker processes ------------------------------------------
    if verbose:
        _info(f"Launching {num_workers} workers...")
    for p in processes:
        p.start()

    # --- Main process: poll workers and update progress ------------------
    try:
        with Progress(
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=Console(file=sys.stdout),
            refresh_per_second=1,
            disable=not verbose,
        ) as progress:
            shots_task = progress.add_task("[cyan]Shots", total=shots_cap)
            errors_task = progress.add_task("[cyan]Errors", total=errors_cap)

            while True:
                if th_stop_event is not None and th_stop_event.is_set():
                    mp_stop_event.set()
                    break
                time.sleep(poll_interval_sec)
                total_shots, total_errors = _read_global_totals(
                    num_workers, shots_arr, errors_arr, locks
                )
                progress.update(
                    shots_task,
                    completed=total_shots,
                )
                progress.update(
                    errors_task,
                    completed=total_errors,
                )
                if total_shots >= shots_cap or total_errors >= errors_cap:
                    mp_stop_event.set()
                    break
                if not all(p.is_alive() for p in processes):
                    raise RuntimeError("One or more workers terminated unexpectedly.")
    finally:
        mp_stop_event.set()
        # Drain the result queue BEFORE joining processes to avoid a deadlock.
        # Workers cannot exit until their queued data is flushed to the pipe,
        # and the pipe buffer is finite. If we join() first, the main process
        # waits for workers to exit while workers wait for the pipe to be read.
        worker_stats: list[BenchmarkStats] = []
        for _ in range(num_workers):
            try:
                worker_stats.append(result_queue.get(timeout=60))
            except Empty:
                break
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()

    # --- Check that workers exited cleanly -------------------------------
    for i, p in enumerate(processes):
        if p.exitcode > 0:
            raise RuntimeError(f"Worker {i} crashed with exit code {p.exitcode}.")
        if p.exitcode < 0 and p.exitcode != -9:
            raise RuntimeError(f"Worker {i} was killed by signal {-p.exitcode}.")

    if len(worker_stats) < num_workers:
        raise RuntimeError(
            f"Only received results from {len(worker_stats)}/{num_workers} workers."
        )

    # --- Aggregate final stats -------------------------------------------
    stats = BenchmarkStats(metadata=metadata)
    for ws in worker_stats:
        stats.merge(ws)
    return stats


def collect_stats(
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    metadata: TaskMetadata,
    *,
    batch_size: int,
    shots_cap: int,
    errors_cap: int,
    num_parallel_workers: int,
    poll_interval_sec: float = 1.0,
    csv_path: Path | str | None = None,
    verbose: bool = True,
    th_stop_event: Optional[threading.Event] = None,
) -> BenchmarkStats:
    """Collect Monte Carlo benchmark statistics for a single task.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to sample from.

    decoder : BenchmarkDecoder
        (Wrapped) decoder to evaluate.

    metadata : TaskMetadata
        Metadata identifying this benchmark task.

    batch_size : int
        Number of shots in one batch.

    shots_cap, errors_cap : int
        The benchmark is considered complete when either the total number of shots
        reaches `shots_cap`, or the number of shots with incorrect observable
        predictions reaches `errors_cap`.

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
    csv_path = Path(csv_path) if csv_path is not None else None

    # --- Resume check ---------------------------------------------------------
    if csv_path is not None:
        csv_stats_list = BenchmarkStats.load_csv(csv_path)
        prev_stats = BenchmarkStats.find_by_metadata(csv_stats_list, metadata)
    else:
        prev_stats = None

    if prev_stats is not None:
        if prev_stats.is_complete(shots_cap, errors_cap):
            if verbose:
                _info("Task already complete.")
                rprint(metadata)
            return prev_stats
        else:
            resuming = True
            shots_cap = shots_cap - prev_stats.shots
            errors_cap = errors_cap - prev_stats.obser_errors
            if verbose:
                _info("Resume from incomplete task.")
                rprint(metadata)
    else:
        resuming = False
        if verbose:
            _info("Start new task.")
            rprint(metadata)

    # --- Run MC loop ----------------------------------------------------------
    if verbose:
        t_start = time.time()

    if num_parallel_workers <= 0:  # serial mode
        stats = _run_serial_collect(
            dem,
            decoder,
            metadata,
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
            metadata,
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
