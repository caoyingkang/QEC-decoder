"""Streaming dataset sampling decoding shots on the fly from a circuit's DEM."""

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from qecdec.circuits import QECCircuit

from .dataset import DecodingDataset

# Shots per stim sampler call; bounds memory while amortizing sampling overhead.
_CHUNK_SIZE = 65536


def sample_decoding_dataset(
    circuit: QECCircuit, *, shots: int, seed: int
) -> DecodingDataset:
    """
    Sample a fixed finite `DecodingDataset` from the circuit's DEM sampler.
    With a fixed seed, this gives a reproducible split (e.g. for validation,
    so metrics are comparable across epochs and runs).

    Parameters
    ----------
        circuit : QECCircuit
            Circuit whose detector error model is sampled.

        shots : int
            Number of shots to sample.

        seed : int
            Seed for the stim sampler.
    """
    sampler = circuit.stim_dem.compile_sampler(seed=seed)
    syndromes, observables, _ = sampler.sample(shots)
    return DecodingDataset(syndromes, observables)


class StreamingDecodingDataset(IterableDataset):
    """
    An `IterableDataset` that samples (syndrome, observable) shots on the fly
    from a circuit's DEM sampler, inside DataLoader workers. Shots follow the
    natural distribution: trivial all-zero-syndrome shots are kept. Items match
    `DecodingDataset`: a pair of int32 tensors of shapes (num_chks,) and
    (num_obsers,).

    Each epoch yields `shots_per_epoch` shots, split evenly across DataLoader
    workers. Seeding: inside a worker, the stream is seeded from the worker
    seed the DataLoader assigns, which varies per worker and per epoch and is
    reproducible under a seeded DataLoader `generator`; in single-process use,
    from `base_seed` and an epoch counter.

    The error rate is settable (for noise curricula): setting `error_rate`
    discards the cached circuit, and the next epoch samples from a circuit
    rebuilt at the new rate (the DEM depends on the error rate). Use
    `persistent_workers=False` so workers are re-created each epoch and pick
    up the change (worker processes hold copies of this dataset).
    """

    def __init__(
        self,
        circuit_factory: Callable[[float], QECCircuit],
        *,
        error_rate: float,
        shots_per_epoch: int,
        base_seed: int = 0,
    ):
        """
        Parameters
        ----------
            circuit_factory : Callable[[float], QECCircuit]
                Builds the circuit for a given error rate. Must be picklable
                for use with multiprocessing DataLoader workers (e.g. a
                module-level function or a `functools.partial` of one).

            error_rate : float
                Initial uniform error rate.

            shots_per_epoch : int
                Number of shots yielded per epoch (across all workers).

            base_seed : int
                Seed for single-process (num_workers=0) streams.
        """
        super().__init__()
        if shots_per_epoch < 1:
            raise ValueError(
                f"shots_per_epoch must be at least 1, but got {shots_per_epoch}"
            )
        self.circuit_factory = circuit_factory
        self.shots_per_epoch = shots_per_epoch
        self.base_seed = base_seed
        self._error_rate = error_rate
        self._circuit: QECCircuit | None = None
        self._epoch = 0

    @property
    def error_rate(self) -> float:
        return self._error_rate

    @error_rate.setter
    def error_rate(self, value: float) -> None:
        if value != self._error_rate:
            self._error_rate = value
            self._circuit = None

    @property
    def circuit(self) -> QECCircuit:
        """The circuit at the current error rate (rebuilt lazily on change)."""
        if self._circuit is None:
            self._circuit = self.circuit_factory(self._error_rate)
        return self._circuit

    def __getstate__(self):
        # Drop the cached circuit: stim-backed objects may not pickle, and
        # worker copies should rebuild at their current error rate anyway.
        return {**self.__dict__, "_circuit": None}

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            shots = self.shots_per_epoch
            seed_seq = np.random.SeedSequence(
                entropy=self.base_seed, spawn_key=(self._epoch,)
            )
            self._epoch += 1
        else:
            base, remainder = divmod(self.shots_per_epoch, worker.num_workers)
            shots = base + (1 if worker.id < remainder else 0)
            seed_seq = np.random.SeedSequence(entropy=worker.seed)
        seed = int(seed_seq.generate_state(1)[0])

        sampler = self.circuit.stim_dem.compile_sampler(seed=seed)
        remaining = shots
        while remaining > 0:
            num = min(_CHUNK_SIZE, remaining)
            syndromes, observables, _ = sampler.sample(num)
            yield from zip(
                torch.from_numpy(syndromes).to(torch.int32),
                torch.from_numpy(observables).to(torch.int32),
            )
            remaining -= num
