from abc import ABC, abstractmethod
from functools import cached_property, total_ordering
from dataclasses import dataclass

import numpy as np
import stim

from .utils import extract_error_mechanisms_from_dem, extract_detector_coords_from_dem
from ..types import (
    Bit2DArray,
    Float1DArray,
    Float2DArray,
)


@total_ordering
@dataclass(frozen=True)
class ErrorMechanism:
    """A dataclass representing an error mechanism.

    Attributes
    ----------
    dets : tuple[int, ...]
        Flipped detectors (in increasing order), nonempty.
    obsers : tuple[int, ...]
        Flipped observables (in increasing order).
    p : float
        Probability of occurrence.
    """

    dets: tuple[int, ...]
    obsers: tuple[int, ...]
    p: float

    def __eq__(self, other) -> bool:
        """
        Two error mechanisms are considered equal if they flip the same set of detectors. In most cases, this
        guarantees that they also flip the same set of observables, because otherwise the fault distance of the
        circuit would be at most 2.
        """
        if not isinstance(other, ErrorMechanism):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        return self.dets == other.dets

    def __lt__(self, other) -> bool:
        if not isinstance(other, ErrorMechanism):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        return self.dets < other.dets


class Experiment(ABC):
    """Abstract base class for QEC experiments.

    Provides properties chkmat, obsmat, prior, etc. from a stim circuit.
    """

    @property
    @abstractmethod
    def circuit(self) -> stim.Circuit:
        """Stim circuit for the experiment."""
        ...

    @cached_property
    def dem(self) -> stim.DetectorErrorModel:
        """Stim detector error model for the experiment."""
        return self.circuit.detector_error_model()

    @cached_property
    def num_detectors(self) -> int:
        """Number of detectors in the circuit."""
        return self.circuit.num_detectors

    @cached_property
    def num_observables(self) -> int:
        """Number of observables in the circuit."""
        return self.circuit.num_observables

    @cached_property
    def error_mechanisms(self) -> list[ErrorMechanism]:
        """(Sorted) list of ErrorMechanism objects."""
        eff2prob = extract_error_mechanisms_from_dem(self.dem)
        emechs: list[ErrorMechanism] = []
        for (dets, obsers), p in eff2prob.items():
            assert len(dets) > 0
            emechs.append(ErrorMechanism(dets, obsers, p))
        emechs.sort()
        return emechs

    @cached_property
    def num_error_mechanisms(self) -> int:
        """Number of error mechanisms."""
        return len(self.error_mechanisms)

    @cached_property
    def chkmat(self) -> Bit2DArray:
        """
        Check matrix, shape=(#detectors, #error_mechanisms), dtype=uint8 ∈ {0,1}.
        (i, j) entry is 1 iff detector i is flipped by error mechanism j.
        """
        chkmat = np.zeros(
            (self.num_detectors, self.num_error_mechanisms), dtype=np.uint8
        )
        for j, e in enumerate(self.error_mechanisms):
            chkmat[e.dets, j] = 1
        return chkmat

    @cached_property
    def obsmat(self) -> Bit2DArray:
        """
        Observable matrix, shape=(#logical_qubits, #error_mechanisms), dtype=uint8 ∈ {0,1}.
        (i, j) entry is 1 iff observable i is flipped by error mechanism j.
        """
        obsmat = np.zeros(
            (self.num_observables, self.num_error_mechanisms), dtype=np.uint8
        )
        for j, e in enumerate(self.error_mechanisms):
            obsmat[e.obsers, j] = 1
        return obsmat

    @cached_property
    def prior(self) -> Float1DArray:
        """
        Prior probabilities for each error mechanism, shape=(#error_mechanisms,), dtype=float64.
        """
        return np.array([e.p for e in self.error_mechanisms])

    @cached_property
    def detector_coords(self) -> Float2DArray:
        """
        Array of detector coordinates, shape=(#detectors, #coordinates), dtype=float64.
        This can be used to visualize the decoding graph.
        """
        return extract_detector_coords_from_dem(self.dem)
