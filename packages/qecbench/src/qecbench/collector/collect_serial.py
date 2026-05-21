import sys
import threading
from typing import Optional

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
from ..utils import _sample


def _collect_stats_serial(
    *,
    dem: stim.DetectorErrorModel,
    decoder: BenchmarkDecoder,
    metadata: TaskMetadata,
    batch_size: int,
    shots_cap: int,
    errors_cap: int,
    verbose: bool,
    th_stop_event: Optional[threading.Event] = None,
) -> TaskStats:
    """Single-process MC collection loop.

    If ``th_stop_event`` is provided, the collection loop checks this event every
    batch and exits early when it is set.
    """
    sampler = dem.compile_sampler()
    stats = TaskStats(metadata=metadata)

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
            result = decoder.decode(syndromes, observables)
            stats.update(
                result.obser_correct_mask,
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
