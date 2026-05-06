"""Memory experiment for the hexagonal (6.6.6 tiling) color code with "superdense"
syndrome extraction circuit.
"""

from typing import Iterable, Literal
from functools import cached_property
from dataclasses import dataclass

import stim

from .base import Experiment


_HEX_OFFSETS: tuple[complex, ...] = (0, 1 + 1j, 2 + 1j, 3, 2 - 1j, 1 - 1j)


@dataclass(frozen=True)
class _Tile:
    """
    Data class for a tile of the color code patch.

    A site is represented by a complex number, where the real part is the horizontal
    coordinate (left to right) and the imaginary part is the vertical coordinate
    (top to bottom).

    Attributes
    ----------
    dq_sites : tuple[complex, ...]
        Sites of data qubits (vertices of the tile).
    xmq_site : complex
        Site of the X-type measurement qubit (left center of the tile).
    zmq_site : complex
        Site of the Z-type measurement qubit (right center of the tile).
    color : int
        Color of the tile. 0: red, 1: green, 2: blue.
    """

    dq_sites: tuple[complex, ...]
    xmq_site: complex
    zmq_site: complex
    color: int


class HexColorCode_Superdense_Memory(Experiment):
    """Memory experiment for the hexagonal (6.6.6 tiling) color code with "superdense"
    syndrome extraction circuit.
    """

    def __init__(
        self,
        *,
        d: int,
        rounds: int,
        basis: Literal["X", "Z"],
        prep_error_rate: float,
        meas_error_rate: float,
        gate1_error_rate: float,
        gate2_error_rate: float,
    ):
        """
        Parameters
        ----------
        d : int
            Code distance. Must be odd and at least 3.
        rounds : int
            Number of rounds of stabilizer measurement. Must be at least 2.
        basis : Literal["X", "Z"]
            Basis of logical state preparation and measurement.
        prep_error_rate : float
            Error rate of state preparation.
        meas_error_rate : float
            Error rate of measurement.
        gate1_error_rate : float
            Error rate of single-qubit gate.
        gate2_error_rate : float
            Error rate of two-qubit gate.
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be an odd number")
        if d < 3:
            raise ValueError("Distance d must be at least 3")
        if rounds < 2:
            raise ValueError("rounds must be at least 2")
        if basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")

        super().__init__()
        self.d = d
        self.rounds = rounds
        self.basis = basis
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate
        self.gate1_error_rate = gate1_error_rate
        self.gate2_error_rate = gate2_error_rate

    @cached_property
    def num_data_qubits(self) -> int:
        """Number of (data) qubits."""
        return 3 * (self.d - 1) * (self.d + 1) // 4 + 1

    @cached_property
    def num_xstabs(self) -> int:
        """Number of X-type stabilizer generators."""
        return 3 * (self.d - 1) * (self.d + 1) // 8

    @cached_property
    def num_zstabs(self) -> int:
        """Number of Z-type stabilizer generators."""
        return 3 * (self.d - 1) * (self.d + 1) // 8

    @cached_property
    def tiles(self) -> list[_Tile]:
        """List of tiles of the color code patch."""
        width = 2 * self.d - 1  # width of the grid holding the data qubits

        def in_bounds(site: complex) -> bool:
            """Whether `site` is inside the triangular patch."""
            if site.imag < 0:
                return False
            if 2 * site.imag > 3 * site.real:
                return False
            if 2 * site.imag > 3 * (width - site.real) - 2:
                return False
            return True

        def make_tile(anchor: complex, color: int) -> _Tile:
            """Make a hexagon tile with leftmost vertex placed at `anchor`."""
            dq_sites = tuple(anchor + o for o in _HEX_OFFSETS if in_bounds(anchor + o))
            return _Tile(
                dq_sites=dq_sites,
                xmq_site=anchor + 1,
                zmq_site=anchor + 2,
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

        assert len(tiles) == self.num_xstabs == self.num_zstabs
        return tiles

    @cached_property
    def dq_sites(self) -> set[complex]:
        """Set of sites of data qubits in the color code patch."""
        dq_sites = set(q for t in self.tiles for q in t.dq_sites)
        assert len(dq_sites) == self.num_data_qubits
        return dq_sites

    @cached_property
    def xmq_sites(self) -> set[complex]:
        """Set of sites of X-type measurement qubits in the color code patch."""
        xmq_sites = set(t.xmq_site for t in self.tiles)
        assert len(xmq_sites) == self.num_xstabs
        return xmq_sites

    @cached_property
    def zmq_sites(self) -> set[complex]:
        """Set of sites of Z-type measurement qubits in the color code patch."""
        zmq_sites = set(t.zmq_site for t in self.tiles)
        assert len(zmq_sites) == self.num_zstabs
        return zmq_sites

    @cached_property
    def all_sites(self) -> set[complex]:
        """Set of sites of all qubits in the color code patch."""
        return self.dq_sites | self.xmq_sites | self.zmq_sites

    @cached_property
    def site2ind(self) -> dict[complex, int]:
        """Dictionary mapping sites of qubits to their indices."""
        sites_sorted = sorted(self.all_sites, key=lambda q: (q.real, q.imag))
        return {q: i for i, q in enumerate(sites_sorted)}

    @cached_property
    def dq_inds(self) -> list[int]:
        """(Sorted) list of indices of data qubits."""
        return sorted(self.site2ind[q] for q in self.dq_sites)

    @cached_property
    def xmq_inds(self) -> list[int]:
        """(Sorted) list of indices of X-type measurement qubits."""
        return sorted(self.site2ind[q] for q in self.xmq_sites)

    @cached_property
    def zmq_inds(self) -> list[int]:
        """(Sorted) list of indices of Z-type measurement qubits."""
        return sorted(self.site2ind[q] for q in self.zmq_sites)

    @cached_property
    def circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()

        # Specify the coordinates of all qubits.
        for q in self.all_sites:
            circuit.append("QUBIT_COORDS", self.site2ind[q], (q.real, q.imag))

        # Prepare the logical state.
        circuit.append(f"R{self.basis}", self.dq_inds)
        circuit += self._stab_meas_circuit()

        return circuit

    def _stab_meas_circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()
        circuit.append("RX", self.xmq_inds)
        circuit.append("RZ", self.zmq_inds)
        circuit.append("TICK")
        circuit += self._cnots(self.xmq_sites, 0, 1)
        circuit.append("TICK")
        return circuit

    def _cnots(
        self,
        offsets: Iterable[complex],
        ctrl: complex,
        tgt: complex,
    ) -> stim.Circuit:
        candidate_pairs = [(o + ctrl, o + tgt) for o in offsets]
        filtered_pairs = [
            pair
            for pair in candidate_pairs
            if pair[0] in self.all_sites and pair[1] in self.all_sites
        ]
        cnot_indices = [self.site2ind[q] for pair in filtered_pairs for q in pair]
        circuit = stim.Circuit()
        circuit.append("CNOT", cnot_indices)
        return circuit
