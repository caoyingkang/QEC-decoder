import multiprocessing
import multiprocessing.synchronize
from queue import Empty
import sys
import threading
import time
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
import stim


from ..decoder_wrapper import BenchmarkDecoder
from ..task import TaskMetadata, TaskStats
from ..utils import _info, _sample


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
    """Worker loop: publish cumulative stats to shared arrays ``shots_arr`` and
    ``errors_arr``; exit when ``mp_stop_event`` is set."""
    stats = TaskStats(metadata=metadata)
    try:
        sampler = dem.compile_sampler(seed=seed)
        while not mp_stop_event.is_set():
            syndromes, observables = _sample(sampler, batch_size)
            result = decoder.decode(syndromes, observables)
            stats.update(
                result.obser_correct_mask,
                result.synd_match_mask,
                result.decoding_iters,
            )
            with locks[worker_id]:
                shots_arr[worker_id] = stats.shots
                errors_arr[worker_id] = stats.obser_errors
    finally:  # Always publish the final stats so the main process will not hang
        result_queue.put(stats)


def _collect_stats_parallel(
    *,
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    metadata: TaskMetadata,
    batch_size: int,
    shots_cap: int,
    errors_cap: int,
    num_workers: int,
    poll_interval_sec: float,
    verbose: bool,
    th_stop_event: Optional[threading.Event],
) -> TaskStats:
    """Multiprocessing MC collection loop.

    The main process polls the worker processes every ``poll_interval_sec``
    seconds to check global completion status.

    If ``th_stop_event`` is provided, the main process checks this event every
    poll interval and exits early when it is set.
    """
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
        worker_stats: list[TaskStats] = []
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
    stats = TaskStats(metadata=metadata)
    for ws in worker_stats:
        stats.merge(ws)
    return stats
