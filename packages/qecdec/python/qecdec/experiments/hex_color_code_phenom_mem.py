"""Memory experiment for the hexagonal (6.6.6 tiling) color code under
phenomenological noise model.
"""

from typing import Literal
from functools import cached_property
from dataclasses import dataclass

import stim

from .base import MemoryExperiment
from ..types import Float2DArray


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


class HexColorCode_Phenom_Memory(MemoryExperiment):
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
        data_qubit_error_rate: float,
        meas_error_rate: float,
    ):
        """
        Parameters
        ----------
        d : int
            Code distance. Must be odd and at least 3.
        rounds : int
            Number of rounds of stabilizer measurement. Must be at least 1.
        basis : Literal["X", "Y", "Z"]
            Basis of logical state preparation and measurement.
        data_qubit_error_rate : float
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

        self.d = d
        self.basis = basis
        self.num_dq = 3 * (d - 1) * (d + 1) // 4 + 1  # number of data qubits
        self.num_stabs = self.num_dq - 1  # number of stabilizer generators

        super().__init__(
            rounds=rounds,
            num_detectors_per_layer=self.num_stabs,
            num_observables=1,
        )

        self.data_qubit_error_rate = data_qubit_error_rate
        self.meas_error_rate = meas_error_rate

        self.tiles = self._build_tiles()

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

        assert 2 * len(tiles) == self.num_stabs
        assert len(set(v for t in tiles for v in t.vertices)) == self.num_dq
        return tiles

    # ==================================================================================
    # Public properties
    # ==================================================================================

    # ==================================================================================
    # Circuit builder
    # ==================================================================================

    @cached_property
    def circuit(self) -> stim.Circuit:
        pass

    @cached_property
    def error_coords(self) -> Float2DArray:
        pass
