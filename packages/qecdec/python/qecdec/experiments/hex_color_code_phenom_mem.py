"""Memory experiment for the hexagonal (6.6.6 tiling) color code under
phenomenological noise model.
"""

from typing import Literal
from functools import cached_property
from dataclasses import dataclass

import stim

from .base import Experiment


_HEX_OFFSETS: tuple[complex, ...] = (0, 1 + 1j, 2 + 1j, 3, 2 - 1j, 1 - 1j)


@dataclass(frozen=True)
class _Tile:
    """
    Data class for a tile of the color code layout.

    Attributes
    ----------
    vertices : tuple[complex, ...]
        Real part: horizontal coordinate, left to right.
        Imaginary part: vertical coordinate, top to down.
    color : int
        Color of the tile. 0: red, 1: green, 2: blue.
    """

    vertices: tuple[complex, ...]
    color: int


class HexColorCode_Phenom_Memory(Experiment):
    """
    Memory experiment for the hexagonal (6.6.6 tiling) color code under
    phenomenological noise model.
    """

    def __init__(
        self,
        *,
        d: int,
        rounds: int,
        basis: Literal["X", "Y", "Z"],
        depolarizing_error_rate: float,
        meas_error_rate: float,
    ):
        """
        Parameters
        ----------
        d : int
            Code distance. Must be odd and at least 3.
        rounds : int
            Number of rounds of stabilizer measurement. Must be at least 1. Every
            round is preceded by depolarizing noise on data qubits. All rounds except
            the last one suffer from measurement errors.
        basis : Literal["X", "Y", "Z"]
            Basis of logical state preparation and measurement.
        depolarizing_error_rate : float
            Error rate of data qubits before each round of stabilizer measurement.
        meas_error_rate : float
            Error rate of measurement.
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be an odd number")
        if d < 3:
            raise ValueError("Distance d must be at least 3")
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        if basis not in ("X", "Y", "Z"):
            raise ValueError("basis must be 'X', 'Y', or 'Z'")

        super().__init__()
        self.d = d
        self.rounds = rounds
        self.basis = basis
        self.depolarizing_error_rate = depolarizing_error_rate
        self.meas_error_rate = meas_error_rate

        self.num_dq = 3 * (d - 1) * (d + 1) // 4 + 1  # number of data qubits
        self.num_stabs = self.num_dq - 1  # number of stabilizer generators

        self.tiles = self._build_tiles()
        assert 2 * len(self.tiles) == self.num_stabs
        dq_sites_set = set(v for t in self.tiles for v in t.vertices)
        assert len(dq_sites_set) == self.num_dq
        self.dq_sites = sorted(dq_sites_set, key=lambda v: (v.real, v.imag))
        self.dq_site2ind = {v: i for i, v in enumerate(self.dq_sites)}

    def _build_tiles(self) -> list[_Tile]:
        width = 2 * self.d - 1  # width of the grid holding the data qubits

        def in_bounds(vertex: complex) -> bool:
            """Whether `vertex` is inside the triangular patch."""
            if vertex.imag < 0:
                return False
            if 2 * vertex.imag > 3 * vertex.real:
                return False
            if 2 * vertex.imag > 3 * (width - vertex.real) - 2:
                return False
            return True

        def make_tile(anchor: complex, color: int) -> _Tile:
            """Make a hexagon tile with leftmost vertex placed at `anchor`."""
            vertices = tuple(anchor + o for o in _HEX_OFFSETS if in_bounds(anchor + o))
            return _Tile(
                vertices=vertices,
                color=color,
            )

        tiles: list[_Tile] = []
        num_periods = (self.d - 1) // 2
        for i in range(num_periods):
            y = 3 * i
            for j in range(num_periods - i):
                x = 2 * i + 4 * j
                red_anchor = x + 1j * y
                tiles.append(make_tile(red_anchor, 0))
                tiles.append(make_tile(red_anchor + 2 + 1j, 1))
                tiles.append(make_tile(red_anchor + 2j, 2))

        return tiles

    @cached_property
    def circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()

        # Specify the coordinates of all data qubits.
        for i, v in enumerate(self.dq_sites):
            circuit.append("QUBIT_COORDS", i, (v.real, v.imag))

        # Noiseless logical state preparation.
        circuit += self._make_circuit_noiseless_logical_meas()
        circuit += self._make_circuit_stab_meas(noisy=False, record_detectors=False)

        # Repeated rounds of depolarizing noise + faulty stabilizer measurement.
        circuit.append(
            stim.CircuitRepeatBlock(
                self.rounds - 1,
                self._make_circuit_depolarizing()
                + self._make_circuit_stab_meas(noisy=True, record_detectors=True),
            )
        )

        # Final round of depolarizing noise + noiseless measurement.
        circuit += self._make_circuit_depolarizing()
        circuit += self._make_circuit_stab_meas(noisy=False, record_detectors=True)
        circuit += self._make_circuit_noiseless_logical_meas()
        return circuit

    def _make_circuit_noiseless_logical_meas(self) -> stim.Circuit:
        circuit = stim.Circuit()
        circuit.append("MPP", self._mpp_targets(self.dq_sites, self.basis))
        circuit.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), 0)
        circuit.append("TICK")
        return circuit

    def _make_circuit_depolarizing(self) -> stim.Circuit:
        circuit = stim.Circuit()
        circuit.append("DEPOLARIZE1", range(self.num_dq), self.depolarizing_error_rate)
        circuit.append("TICK")
        return circuit

    def _make_circuit_stab_meas(
        self, noisy: bool, record_detectors: bool
    ) -> stim.Circuit:
        circuit = stim.Circuit()
        for stab_basis in ("X", "Z"):
            for tile in self.tiles:
                circuit.append(
                    "MPP",
                    self._mpp_targets(tile.vertices, stab_basis),
                    self.meas_error_rate if noisy else None,
                )
        if record_detectors:
            offset = 0
            for stab_basis in ("X", "Z"):
                for tile in self.tiles:
                    center = sum(tile.vertices) / len(tile.vertices)
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(-self.num_stabs + offset),
                            stim.target_rec(-self.num_stabs * 2 + offset),
                        ],
                        (center.real, center.imag, 0),
                    )
                    offset += 1
            circuit.append("SHIFT_COORDS", [], (0, 0, 1))

        circuit.append("TICK")
        return circuit

    def _mpp_targets(
        self, sites: list[complex], pauli: Literal["X", "Y", "Z"]
    ) -> list[stim.GateTarget]:
        indices = sorted(self.dq_site2ind[v] for v in sites)
        targets: list[stim.GateTarget] = []
        for i in indices:
            if len(targets) > 0:
                targets.append(stim.target_combiner())
            targets.append(stim.target_pauli(i, pauli))
        return targets
