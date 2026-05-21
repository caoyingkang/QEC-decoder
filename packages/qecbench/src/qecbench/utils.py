import numpy as np
from rich import print as rprint
import stim

from qecdec.types import Bit2DArray


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
