from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar
from typing_extensions import Self

import numpy as np
import stim

from ..types import Bit2DArray, Float1DArray, Float2DArray
from .errmech import ErrorMechanism
from .utils import _extract_detector_coords_from_dem, _extract_error_mechanisms_from_dem


class QECCircuit(ABC):
    """Abstract base class for QEC circuits.

    Wraps a ``stim.Circuit`` and provides helpful properties including
    ``chkmat``, ``obsmat``, ``prior``, etc.
    """

    registry: ClassVar[dict[str, type["QECCircuit"]]] = {}

    def __init_subclass__(cls, registry_name: str | None = None) -> None:
        # Only register subclasses that set ``registry_name``.
        if registry_name is not None:
            if registry_name in QECCircuit.registry:
                raise ValueError(
                    f"QECCircuit registry_name {registry_name!r} is already assigned."
                )
            QECCircuit.registry[registry_name] = cls

    def __init__(self, circuit: stim.Circuit) -> None:
        self.stim_circuit = circuit

    @classmethod
    @abstractmethod
    def with_uniform_error_rate(
        cls,
        error_rate: float,
        **kwargs,
    ) -> Self:
        """Alternative constructor with all fault locations having the
        same error rate.
        """
        ...

    @property
    def num_detectors(self) -> int:
        """Number of detectors in the circuit."""
        return self.stim_circuit.num_detectors

    @property
    def num_observables(self) -> int:
        """Number of observables in the circuit."""
        return self.stim_circuit.num_observables

    @cached_property
    def stim_dem(self) -> stim.DetectorErrorModel:
        """Return ``stim.DetectorErrorModel`` object."""
        return self.stim_circuit.detector_error_model()

    @cached_property
    def error_mechanisms(self) -> list[ErrorMechanism]:
        """List of ``ErrorMechanism`` objects, sorted by the tuple of flipped detectors."""
        eff2prob = _extract_error_mechanisms_from_dem(self.stim_dem)
        return sorted(
            (
                ErrorMechanism(dets=dets, obsers=obsers, p=p)
                for (dets, obsers), p in eff2prob.items()
            ),
            key=lambda e: e.dets,
        )

    @property
    def num_error_mechanisms(self) -> int:
        """Number of error mechanisms."""
        return len(self.error_mechanisms)

    @cached_property
    def chkmat(self) -> Bit2DArray:
        """
        Check matrix, shape=(num_detectors, num_error_mechanisms).
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
        Observable matrix, shape=(num_observables, num_error_mechanisms).
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
        Prior probabilities for each error mechanism, shape=(num_error_mechanisms,).
        """
        return np.array([e.p for e in self.error_mechanisms])

    @cached_property
    def detector_coords(self) -> Float2DArray:
        """
        Array of detector coordinates, shape=(num_detectors, #coordinates).
        This can be used to visualize the decoding graph.
        """
        return _extract_detector_coords_from_dem(self.stim_dem)


# Module-level alias for the class-attribute registry. This points to the
# same underlying object, so updates from `__init_subclass__` flow through to
# `from qecdec.circuits import CIRCUITS_REGISTRY` callers.
CIRCUITS_REGISTRY = QECCircuit.registry
