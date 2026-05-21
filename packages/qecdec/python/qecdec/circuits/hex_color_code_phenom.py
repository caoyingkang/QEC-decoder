from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Literal

import stim

from .base import QECCircuit


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
    sites : tuple[complex, ...]
        Sites of data qubits (vertices of the tile).
    color : int
        Color of the tile. 0: red, 1: green, 2: blue.
    """

    sites: tuple[complex, ...]
    color: int


class HexColorCode_Phenom(QECCircuit, registry_name="HexColorCode_Phenom"):
    """Memory circuit for the hexagonal (6.6.6 tiling) color code under
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
            Number of rounds of stabilizer measurement. Must be at least 2. Every
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
        if rounds < 2:
            raise ValueError("rounds must be at least 2")
        if basis not in ("X", "Y", "Z"):
            raise ValueError("basis must be 'X', 'Y', or 'Z'")

        self.d = d
        self.rounds = rounds
        self.basis = basis
        self.depolarizing_error_rate = depolarizing_error_rate
        self.meas_error_rate = meas_error_rate

        circuit = self._build_circuit()
        super().__init__(circuit)

    @classmethod
    def with_uniform_error_rate(
        cls,
        error_rate: float,
        *,
        d: int,
        rounds: int,
        basis: Literal["X", "Y", "Z"],
    ) -> "HexColorCode_Phenom":
        return HexColorCode_Phenom(
            d=d,
            rounds=rounds,
            basis=basis,
            depolarizing_error_rate=error_rate,
            meas_error_rate=error_rate,
        )

    @cached_property
    def num_qubits(self) -> int:
        """Number of (data) qubits."""
        return 3 * (self.d - 1) * (self.d + 1) // 4 + 1

    @cached_property
    def num_stabs(self) -> int:
        """Number of stabilizer generators."""
        return 3 * (self.d - 1) * (self.d + 1) // 4

    @cached_property
    def _tiles(self) -> list[_Tile]:
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
            sites = tuple(anchor + o for o in _HEX_OFFSETS if in_bounds(anchor + o))
            return _Tile(
                sites=sites,
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
        return tiles

    @cached_property
    def sites(self) -> set[complex]:
        """Set of sites of (data) qubits in the color code patch."""
        sites = set(q for t in self._tiles for q in t.sites)
        assert len(sites) == self.num_qubits
        return sites

    @cached_property
    def sites_sorted(self) -> list[complex]:
        """Sorted list of sites of (data) qubits in the color code patch."""
        return sorted(self.sites, key=lambda q: (q.real, q.imag))

    @cached_property
    def site2ind(self) -> dict[complex, int]:
        """Dictionary mapping sites of qubits to their indices."""
        return {q: i for i, q in enumerate(self.sites_sorted)}

    def _build_circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()

        # Specify the coordinates of all data qubits.
        for i, q in enumerate(self.sites_sorted):
            circuit.append("QUBIT_COORDS", i, (q.real, q.imag))

        # Noiseless logical state preparation.
        circuit += self._noiseless_logical_meas()
        circuit += self._stab_meas_circuit(noisy=False, record_detectors=False)

        # Repeated rounds of depolarizing noise + faulty stabilizer measurement.
        circuit.append(
            stim.CircuitRepeatBlock(
                self.rounds - 1,
                self._depolarizing_noise()
                + self._stab_meas_circuit(noisy=True, record_detectors=True),
            )
        )

        # Final round of depolarizing noise + noiseless measurement.
        circuit += self._depolarizing_noise()
        circuit += self._stab_meas_circuit(noisy=False, record_detectors=True)
        circuit += self._noiseless_logical_meas()
        return circuit

    def _noiseless_logical_meas(self) -> stim.Circuit:
        circuit = stim.Circuit()
        circuit.append("MPP", self._mpp_targets(self.sites, self.basis))
        circuit.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), 0)
        circuit.append("TICK")
        return circuit

    def _depolarizing_noise(self) -> stim.Circuit:
        circuit = stim.Circuit()
        circuit.append(
            "DEPOLARIZE1", range(self.num_qubits), self.depolarizing_error_rate
        )
        circuit.append("TICK")
        return circuit

    def _stab_meas_circuit(
        self, *, noisy: bool, record_detectors: bool
    ) -> stim.Circuit:
        circuit = stim.Circuit()
        for stab_basis in ("X", "Z"):
            for tile in self._tiles:
                circuit.append(
                    "MPP",
                    self._mpp_targets(tile.sites, stab_basis),
                    self.meas_error_rate if noisy else None,
                )
        if record_detectors:
            offset = 0
            for stab_basis in ("X", "Z"):
                for tile in self._tiles:
                    center = sum(tile.sites) / len(tile.sites)
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
        self, sites: Iterable[complex], pauli: Literal["X", "Y", "Z"]
    ) -> list[stim.GateTarget]:
        indices = sorted(self.site2ind[q] for q in sites)
        targets: list[stim.GateTarget] = []
        for i in indices:
            if len(targets) > 0:
                targets.append(stim.target_combiner())
            targets.append(stim.target_pauli(i, pauli))
        return targets
