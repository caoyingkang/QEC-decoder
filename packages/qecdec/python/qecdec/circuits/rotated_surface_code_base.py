from abc import abstractmethod
from functools import cached_property
from typing import Literal

import numpy as np
import stim

from ..types import Bool2DArray, Bool3DArray, Int2DArray
from .base import QECCircuit


class RotatedSurfaceCodeBase(QECCircuit):
    """Shared lattice geometry for rotated surface code memory circuits.

    Subclasses provide the noise model: they set their noise parameters, then call
    ``super().__init__(d=d, rounds=rounds, basis=basis)``, which sets up the lattice
    and invokes the subclass's ``_build_circuit()``.

    Besides the lattice attributes used to build the ``stim.Circuit``, this class
    exposes the geometry of the detectors on a dense spacetime grid (used by e.g. CNN
    decoders): the unique x (resp. y) coordinates of the in-basis checks are
    ranked to dense column (resp. row) indices, and each time slice of detectors
    forms one grid layer. Detectors of one basis occupy a checkerboard subset of the
    spatial grid; the remaining sites are free and are where data-qubit features
    live in a CNN readout.
    """

    def __init__(self, *, d: int, rounds: int, basis: Literal["X", "Z"]):
        """
        Parameters
        ----------
            d : int
                Code distance. Must be odd and at least 3.
            rounds : int
                Number of rounds of stabilizer measurement. Must be at least 2.
            basis : Literal['X', 'Z']
                Basis of logical state preparation and measurement. If basis='X'
                (resp. 'Z'), then we will use X-type (resp. Z-type) stabilizer
                measurement outcomes to correct Pauli Z (resp. X) errors.
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be an odd number")
        if d < 3:
            raise ValueError("Distance d must be at least 3")
        if rounds < 2:
            raise ValueError("rounds must be at least 2")
        if basis not in ["X", "Z"]:
            raise ValueError("Basis must be 'X' or 'Z'")

        self.d = d
        self.rounds = rounds
        self.basis = basis

        self.w = 2 * d + 1  # width of the grid holding the qubits
        self.num_dq = d * d  # number of data qubits
        self.num_xmq = (d * d - 1) // 2  # number of X-type measure qubits
        self.num_zmq = (d * d - 1) // 2  # number of Z-type measure qubits
        self.num_mq = self.num_xmq + self.num_zmq  # total number of measure qubits
        self.num_qubits = self.num_dq + self.num_mq  # total number of physical qubits

        # Lattice site coordinates of data qubits and measure qubits.
        self.dq_coos = frozenset(
            (x, y)
            for x in range(self.w)
            for y in range(self.w)
            if self._is_data_qubit_coord(x, y)
        )
        self.xmq_coos = frozenset(
            (x, y)
            for x in range(self.w)
            for y in range(self.w)
            if self._is_x_meas_qubit_coord(x, y)
        )
        self.zmq_coos = frozenset(
            (x, y)
            for x in range(self.w)
            for y in range(self.w)
            if self._is_z_meas_qubit_coord(x, y)
        )
        self.mq_coos = self.xmq_coos | self.zmq_coos
        assert len(self.dq_coos) == self.num_dq
        assert len(self.xmq_coos) == self.num_xmq
        assert len(self.zmq_coos) == self.num_zmq
        assert len(self.mq_coos) == self.num_mq

        # Lattice site indices of data qubits and measure qubits (sorted in ascending order).
        self.dq_inds = sorted(self._coo2ind(*coo) for coo in self.dq_coos)
        self.xmq_inds = sorted(self._coo2ind(*coo) for coo in self.xmq_coos)
        self.zmq_inds = sorted(self._coo2ind(*coo) for coo in self.zmq_coos)
        self.mq_inds = sorted(self.xmq_inds + self.zmq_inds)

        circuit = self._build_circuit()
        super().__init__(circuit)

    @abstractmethod
    def _build_circuit(self) -> stim.Circuit:
        """Build the ``stim.Circuit``."""
        ...

    # ----------------------------------------------------------------------------------------
    # CNN-grid geometry
    # ----------------------------------------------------------------------------------------

    @cached_property
    def basis_check_coos(self) -> list[tuple[int, int]]:
        """
        Lattice (x, y) coordinates of the in-basis checks, in detector order:
        ascending lattice site index, exactly the order detectors are appended
        within each round in ``_build_circuit``.
        """
        check_coos = self.zmq_coos if self.basis == "Z" else self.xmq_coos
        return sorted(check_coos, key=lambda coo: self._coo2ind(*coo))

    @cached_property
    def grid_shape(self) -> tuple[int, int, int]:
        """
        Shape (num_layers, height, width) of the dense spacetime grid of detectors:
        a layer is one time slice of detectors, so num_layers = ``rounds``
        measurement rounds + 1 final-measurement slice; height (resp. width) is
        the number of unique check y (resp. x) coordinates.
        """
        xs = {x for x, _ in self.basis_check_coos}
        ys = {y for _, y in self.basis_check_coos}
        return self.rounds + 1, len(ys), len(xs)

    @cached_property
    def _check_grid_sites(self) -> list[tuple[int, int]]:
        """Spatial grid site (row, col) of each basis check, in detector order."""
        x_rank = {
            x: i for i, x in enumerate(sorted({x for x, _ in self.basis_check_coos}))
        }
        y_rank = {
            y: i for i, y in enumerate(sorted({y for _, y in self.basis_check_coos}))
        }
        return [(y_rank[y], x_rank[x]) for x, y in self.basis_check_coos]

    @cached_property
    def detector_grid_sites(self) -> Int2DArray:
        """
        Grid site of each detector, shape=(num_detectors, 3). Each row is
        (layer, row, col). Detector index = layer * num_basis_checks + k, with k
        indexing ``basis_check_coos``.
        """
        num_layers, _, _ = self.grid_shape
        sites = np.array(
            [
                (layer, row, col)
                for layer in range(num_layers)
                for row, col in self._check_grid_sites
            ],
            dtype=np.int64,
        )
        assert len(sites) == self.num_detectors
        return sites

    @cached_property
    def detector_site_mask(self) -> Bool2DArray:
        """Spatial grid sites hosting a detector, shape=(height, width)."""
        _, height, width = self.grid_shape
        mask = np.zeros((height, width), dtype=np.bool_)
        for row, col in self._check_grid_sites:
            mask[row, col] = True
        return mask

    @cached_property
    def observable_grid_masks(self) -> Bool3DArray:
        """
        Per-observable data-qubit support masks, shape=(num_observables, height,
        width). Marks the detector-free grid sites diagonally adjacent (on the
        qubit lattice) to the data qubits in the observable's support.
        """
        # Support data qubits of the single logical observable, matching the
        # OBSERVABLE_INCLUDE targets in `_build_circuit`: the first d entries of
        # `dq_inds` (bottom row, y=1) for basis='Z', every d-th entry (left
        # column, x=1) for basis='X'.
        if self.basis == "Z":
            support_inds = self.dq_inds[: self.d]
        else:
            support_inds = self.dq_inds[:: self.d]
        support_coos = [self._ind2coo(i) for i in support_inds]

        check_coo_set = set(self.basis_check_coos)
        x_rank = {
            x: i for i, x in enumerate(sorted({x for x, _ in self.basis_check_coos}))
        }
        y_rank = {
            y: i for i, y in enumerate(sorted({y for _, y in self.basis_check_coos}))
        }

        _, height, width = self.grid_shape
        masks = np.zeros((self.num_observables, height, width), dtype=np.bool_)
        for x, y in support_coos:
            for cx, cy in (
                (x - 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y - 1),
                (x + 1, y + 1),
            ):
                if (cx, cy) not in check_coo_set and cx in x_rank and cy in y_rank:
                    masks[0, y_rank[cy], x_rank[cx]] = True
        assert masks.any(axis=(1, 2)).all()
        return masks

    # ----------------------------------------------------------------------------------------
    # Lattice helpers
    # ----------------------------------------------------------------------------------------

    def _is_data_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of a data qubit."""
        return 0 <= x < self.w and 0 <= y < self.w and x % 2 == 1 and y % 2 == 1

    def _is_x_meas_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of an X-type measure qubit."""
        return (
            2 <= x < self.w - 2
            and 0 <= y < self.w
            and x % 2 == 0
            and y % 2 == 0
            and (x + y) % 4 == 2
        )

    def _is_z_meas_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of a Z-type measure qubit."""
        return (
            0 <= x < self.w
            and 2 <= y < self.w - 2
            and x % 2 == 0
            and y % 2 == 0
            and (x + y) % 4 == 0
        )

    def _coo2ind(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to lattice site index."""
        if (x + y) % 2 == 1:
            raise ValueError("Not a valid lattice site")
        return (y // 2) * self.w + x

    def _ind2coo(self, i: int) -> tuple[int, int]:
        """Convert lattice site index to (x, y) coordinates."""
        x = i % self.w
        y = 2 * (i // self.w) + (x % 2)
        return x, y
