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
    start_layer : int
        The first layer in the decoding graph to find a flipped detector.
    end_layer : int
        The last layer plus one in the decoding graph to find a flipped detector.
    """

    dets: tuple[int, ...]
    obsers: tuple[int, ...]
    p: float
    start_layer: int
    end_layer: int

    def __eq__(self, other) -> bool:
        """
        Two error mechanisms are considered equal if they flip the same set of detectors. In most cases, this
        guarantees that they also flip the same set of observables, because otherwise the fault distance of the
        circuit would be at most 2.
        """
        if not isinstance(other, ErrorMechanism):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        # The following guarantees that self.start_layer == other.start_layer and self.end_layer == other.end_layer
        return self.dets == other.dets

    def __lt__(self, other) -> bool:
        """
        To sort error mechanisms, we first compare `start_layer`, then `end_layer`, and finally `dets`.
        """
        if not isinstance(other, ErrorMechanism):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        if self.start_layer != other.start_layer:
            return self.start_layer < other.start_layer
        elif self.end_layer != other.end_layer:
            return self.end_layer < other.end_layer
        else:
            return self.dets < other.dets


class MemoryExperiment(ABC):
    """Abstract base class for memory experiments."""

    def __init__(
        self,
        rounds: int,
        num_detectors_per_layer: int,
        num_observables: int,
    ):
        """
        Parameters
        ----------
        rounds : int
            Number of rounds of stabilizer measurement.
        num_detectors_per_layer : int
            Number of layers of detectors in the decoding graph.
        num_observables : int
            Number of logical observables.
        """
        self.rounds = rounds
        self.layers = rounds + 1
        self.num_detectors_per_layer = num_detectors_per_layer
        self.num_detectors = self.layers * num_detectors_per_layer
        self.num_observables = num_observables

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
    def eid2emech(self) -> dict[int, ErrorMechanism]:
        """Dictionary mapping error ids to ErrorMechanism objects."""
        eff2prob = extract_error_mechanisms_from_dem(self.dem)

        emechs: list[ErrorMechanism] = []
        for (dets, obsers), p in eff2prob.items():
            assert len(dets) > 0
            start_layer = dets[0] // self.num_detectors_per_layer
            end_layer = dets[-1] // self.num_detectors_per_layer + 1
            emechs.append(ErrorMechanism(dets, obsers, p, start_layer, end_layer))
        emechs.sort()

        return {i: e for i, e in enumerate(emechs)}

    @cached_property
    def num_error_mechanisms(self) -> int:
        """Number of error mechanisms."""
        return len(self.eid2emech)

    @cached_property
    def chkmat(self) -> Bit2DArray:
        """
        Check matrix, shape=(#detectors, #error_mechanisms), dtype=uint8 ∈ {0,1}.
        (i, j) entry is 1 iff detector i is flipped by error mechanism j.
        """
        chkmat = np.zeros(
            (self.num_detectors, self.num_error_mechanisms), dtype=np.uint8
        )
        for j, e in self.eid2emech.items():
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
        for j, e in self.eid2emech.items():
            obsmat[e.obsers, j] = 1
        return obsmat

    @cached_property
    def prior(self) -> Float1DArray:
        """
        Prior probabilities for each error mechanism, shape=(#error_mechanisms,), dtype=float64.
        """
        prior = np.zeros(self.num_error_mechanisms)
        for j, e in self.eid2emech.items():
            prior[j] = e.p
        return prior

    @cached_property
    def detector_coords(self) -> Float2DArray:
        """
        Array of detector coordinates, shape=(#detectors, #coordinates), dtype=float64.
        This can be used to visualize the decoding graph.
        """
        return extract_detector_coords_from_dem(self.dem)

    @property
    @abstractmethod
    def error_coords(self) -> Float2DArray:
        """
        Array of error coordinates, shape=(#error_mechanisms, #coordinates), dtype=float64.
        This can be used to visualize the decoding graph.
        """
        ...
