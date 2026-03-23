"""Monte Carlo collection loop — replacement for sinter.collect."""
import multiprocessing
from multiprocessing.synchronize import Lock, Event
from multiprocessing import Process, Queue
from queue import Empty
import time
from pathlib import Path

import humanize
import numpy as np
import stim

from .decoder import BenchmarkDecoder
from .stats import BenchmarkStats, TaskMetadata
from .types import Bit2DArray


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


def _print_progress(  # TODO: also show speed in real time (shots/s) and ETA?
    elapsed: float,
    shots_remain: int,
    errors_remain: int,
) -> None:
    """Print progress to stdout."""
    parts = [
        f"Elapsed: {elapsed:.1f}s",
        f"Shots left: {shots_remain:,}",
        f"Errors left: {errors_remain:,}",
    ]
    line = ", ".join(parts)
    print(line, end="\r", flush=True)  # TODO: will this overwrite the previous line properly?


def _run_serial_collect(
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    metadata: TaskMetadata,
    *,
    batch_size: int,
    shots_cap: int,
    errors_cap: int,
    verbose: bool,
) -> BenchmarkStats:
    """Single-process MC collection loop."""
    sampler = dem.compile_sampler()
    stats = BenchmarkStats(metadata=metadata)

    t_start = time.time()
    while not stats.is_complete(shots_cap, errors_cap):
        syndromes, observables = _sample(sampler, batch_size)
        result = decoder.decode(syndromes)
        obser_correct_mask = np.all(result.obser_pred == observables, axis=1)
        stats.update(
            obser_correct_mask,
            result.synd_match_mask,
            result.decoding_iters,
        )
        if verbose:  # TODO: less frequently: once per 1s?
            elapsed = time.time() - t_start
            _print_progress(
                elapsed,
                max(0, shots_cap - stats.shots),
                max(0, errors_cap - stats.obser_errors),
            )

    if verbose:
        print()
    return stats


def _read_global_totals(
    num_workers: int,
    shots_arr,
    errors_arr,
    locks: list[Lock],
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
    locks: list[Lock],
    stop_event: Event,
    result_queue: Queue[BenchmarkStats],
) -> None:
    """Worker loop: publish cumulative stats to shared arrays `shots_arr` and 
    `errors_arr`; exit when `stop_event` is set."""
    stats = BenchmarkStats(metadata=metadata)
    try:
        sampler = dem.compile_sampler(seed=seed)
        while not stop_event.is_set():
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
) -> BenchmarkStats:
    """Multi-processing MC collection loop."""
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, but got {num_workers}.")
    if poll_interval_sec <= 0:
        raise ValueError(f"poll_interval_sec must be positive, but got {poll_interval_sec}.")

    # lock=False because each worker will write to its own entry
    shots_arr = multiprocessing.Array('q', num_workers, lock=False)
    errors_arr = multiprocessing.Array('q', num_workers, lock=False)
    locks = [Lock() for _ in range(num_workers)]
    stop_event = Event()
    result_queue: Queue[BenchmarkStats] = Queue()

    if verbose:
        print(f"[info] Launching {num_workers} workers...")

    t_start = time.time()
    rng = np.random.default_rng()
    seeds = rng.integers(2**31, size=num_workers)
    processes: list[Process] = []
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
            stop_event,
            result_queue,
        )
        p = Process(target=_worker_loop, args=args)
        p.start()
        processes.append(p)

    try:
        while True:
            time.sleep(poll_interval_sec)
            total_shots, total_errors = _read_global_totals(
                num_workers, shots_arr, errors_arr, locks
            )
            if verbose:
                elapsed = time.time() - t_start
                _print_progress(
                    elapsed,
                    max(0, shots_cap - total_shots),
                    max(0, errors_cap - total_errors),
                )
            if total_shots >= shots_cap or total_errors >= errors_cap:
                stop_event.set()
                break
            if not all(p.is_alive() for p in processes):
                raise RuntimeError("One or more workers terminated unexpectedly.")
    finally:
        if verbose:
            print()
        stop_event.set()
        for p in processes:
            p.join()

    # Check that workers exited cleanly
    for i, p in enumerate(processes):
        if p.exitcode > 0:
            raise RuntimeError(f"Worker {i} crashed with exit code {p.exitcode}.")
        if p.exitcode < 0:
            raise RuntimeError(f"Worker {i} was killed by signal {-p.exitcode}.")

    stats = BenchmarkStats(metadata=metadata)
    for _ in range(num_workers):
        try:
            stats.merge(result_queue.get(timeout=30))
        except Empty:
            raise RuntimeError("Didn't receive result from workers within 30 seconds timeout.")
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
                print(f"[skip] Already complete for task: {metadata}")
            return prev_stats
        else:
            resuming = True
            shots_cap = shots_cap - prev_stats.shots
            errors_cap = errors_cap - prev_stats.obser_errors
            if verbose:
                print(f"[resume] Found incomplete task: {metadata}")
    else:
        resuming = False
        if verbose:
            print(f"[start] New task: {metadata}")

    # --- Run MC loop ----------------------------------------------------------
    if verbose:
        t_start = time.time()

    if num_parallel_workers <= 0:  # serial mode
        stats = _run_serial_collect(
            dem, decoder, metadata,
            batch_size=batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
            verbose=verbose,
        )
    else:  # multiprocessing mode
        stats = _run_parallel_collect(
            dem, decoder, metadata,
            batch_size=batch_size,
            shots_cap=shots_cap,
            errors_cap=errors_cap,
            num_workers=num_parallel_workers,
            poll_interval_sec=poll_interval_sec,
            verbose=verbose,
        )

    if verbose:
        elapsed = time.time() - t_start
        elapsed_str = humanize.precisedelta(elapsed)
        speed_str = f" ({stats.shots / elapsed:.1f} shots/s)" if elapsed > 10.0 else ""
        print(f"[done] Task completed. Collected {stats.shots:,} shots in {elapsed_str}{speed_str}.")

    # Merge with previous partial stats if resuming
    if resuming:
        stats.merge(prev_stats)

    # --- Save -----------------------------------------------------------------
    if csv_path is not None:
        BenchmarkStats.upsert(csv_stats_list, stats)
        BenchmarkStats.save_csv(csv_stats_list, csv_path)  # TODO: should I add a lock to prevent concurrent writes?

    return stats
