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


class HexColorCode_Superdense(QECCircuit, registry_name="HexColorCode_Superdense"):
    """Memory circuit for the hexagonal (6.6.6 tiling) color code with "superdense"
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

        self.d = d
        self.rounds = rounds
        self.basis = basis
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate
        self.gate1_error_rate = gate1_error_rate
        self.gate2_error_rate = gate2_error_rate

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
    ) -> "HexColorCode_Superdense":
        return HexColorCode_Superdense(
            d=d,
            rounds=rounds,
            basis=basis,
            prep_error_rate=error_rate,
            meas_error_rate=error_rate,
            gate1_error_rate=error_rate,
            gate2_error_rate=error_rate,
        )

    @cached_property
    def num_data_qubits(self) -> int:
        """Number of data qubits."""
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
    def num_stabs(self) -> int:
        """Number of stabilizer generators."""
        return self.num_xstabs + self.num_zstabs

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
        dq_sites = set(q for t in self._tiles for q in t.dq_sites)
        assert len(dq_sites) == self.num_data_qubits
        return dq_sites

    @cached_property
    def dq_sites_sorted(self) -> list[complex]:
        """Sorted list of sites of data qubits in the color code patch."""
        return sorted(self.dq_sites, key=lambda q: (q.real, q.imag))

    @cached_property
    def dq_site2relind(self) -> dict[complex, int]:
        """Dictionary mapping sites of data qubits to their relative indices
        in the list ``dq_sites_sorted``."""
        return {q: i for i, q in enumerate(self.dq_sites_sorted)}

    @cached_property
    def xmq_sites(self) -> set[complex]:
        """Set of sites of X-type measurement qubits in the color code patch."""
        xmq_sites = set(t.xmq_site for t in self._tiles)
        assert len(xmq_sites) == self.num_xstabs
        return xmq_sites

    @cached_property
    def xmq_sites_sorted(self) -> list[complex]:
        """Sorted list of sites of X-type measurement qubits in the color code patch."""
        return sorted(self.xmq_sites, key=lambda q: (q.real, q.imag))

    @cached_property
    def xmq_site2relind(self) -> dict[complex, int]:
        """Dictionary mapping sites of X-type measurement qubits to their relative indices
        in the list ``xmq_sites_sorted``."""
        return {q: i for i, q in enumerate(self.xmq_sites_sorted)}

    @cached_property
    def zmq_sites(self) -> set[complex]:
        """Set of sites of Z-type measurement qubits in the color code patch."""
        zmq_sites = set(t.zmq_site for t in self._tiles)
        assert len(zmq_sites) == self.num_zstabs
        return zmq_sites

    @cached_property
    def zmq_sites_sorted(self) -> list[complex]:
        """Sorted list of sites of Z-type measurement qubits in the color code patch."""
        return sorted(self.zmq_sites, key=lambda q: (q.real, q.imag))

    @cached_property
    def zmq_site2relind(self) -> dict[complex, int]:
        """Dictionary mapping sites of Z-type measurement qubits to their relative indices
        in the list ``zmq_sites_sorted``."""
        return {q: i for i, q in enumerate(self.zmq_sites_sorted)}

    @cached_property
    def all_sites(self) -> set[complex]:
        """Set of sites of all qubits in the color code patch."""
        return self.dq_sites | self.xmq_sites | self.zmq_sites

    @cached_property
    def all_sites_sorted(self) -> list[complex]:
        """Sorted list of sites of all qubits in the color code patch."""
        return sorted(self.all_sites, key=lambda q: (q.real, q.imag))

    @property
    def ind2site(self) -> list[complex]:
        """Alternative name for ``all_sites_sorted``. A list functioning as a mapping from
        qubit indices to their sites."""
        return self.all_sites_sorted

    @cached_property
    def site2ind(self) -> dict[complex, int]:
        """Dictionary mapping sites of qubits to their indices."""
        return {q: i for i, q in enumerate(self.ind2site)}

    @cached_property
    def dq_inds(self) -> list[int]:
        """(Sorted) list of indices of data qubits."""
        dq_inds = [self.site2ind[q] for q in self.dq_sites_sorted]
        assert dq_inds == sorted(dq_inds)
        return dq_inds

    @cached_property
    def xmq_inds(self) -> list[int]:
        """(Sorted) list of indices of X-type measurement qubits."""
        xmq_inds = [self.site2ind[q] for q in self.xmq_sites_sorted]
        assert xmq_inds == sorted(xmq_inds)
        return xmq_inds

    @cached_property
    def zmq_inds(self) -> list[int]:
        """(Sorted) list of indices of Z-type measurement qubits."""
        zmq_inds = [self.site2ind[q] for q in self.zmq_sites_sorted]
        assert zmq_inds == sorted(zmq_inds)
        return zmq_inds

    @cached_property
    def mq_ind2tile(self) -> dict[int, _Tile]:
        """Dictionary mapping indices of measurement qubits to their tiles."""
        return {
            self.site2ind[q]: t for t in self._tiles for q in [t.xmq_site, t.zmq_site]
        }

    def _build_circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()

        # Specify the coordinates of all qubits.
        for i, q in enumerate(self.ind2site):
            circuit.append("QUBIT_COORDS", i, (q.real, q.imag))

        # Prepare the logical state.
        first_round = stim.Circuit()
        first_round.append(f"R{self.basis}", self.dq_inds)
        first_round += self._stab_meas_circuit(which_round="first")

        circuit += first_round

        # Middle syndrome extraction rounds.
        if self.rounds > 2:
            circuit += (self.rounds - 2) * self._stab_meas_circuit(which_round="middle")

        # Last round + logical measurement.
        # TODO

        return circuit

    def _stab_meas_circuit(
        self, *, which_round: Literal["first", "middle", "last"]
    ) -> stim.Circuit:
        circuit = stim.Circuit()
        circuit.append("RX", self.xmq_inds)
        circuit.append("RZ", self.zmq_inds)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, 0, 1)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, 1j, 0)
        self._add_cnots(circuit, self.zmq_sites_sorted, 1j, 0)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, -1, 0)
        self._add_cnots(circuit, self.zmq_sites_sorted, 1, 0)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, -1j, 0)
        self._add_cnots(circuit, self.zmq_sites_sorted, -1j, 0)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, 0, 1j)
        self._add_cnots(circuit, self.zmq_sites_sorted, 0, 1j)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, 0, -1)
        self._add_cnots(circuit, self.zmq_sites_sorted, 0, 1)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, 0, -1j)
        self._add_cnots(circuit, self.zmq_sites_sorted, 0, -1j)
        circuit.append("TICK")
        self._add_cnots(circuit, self.xmq_sites_sorted, 0, 1)
        circuit.append("TICK")
        circuit.append("MX", self.xmq_inds)
        circuit.append("MZ", self.zmq_inds)
        self._add_detectors(circuit, which_round=which_round)
        circuit.append("TICK")
        return circuit

    def _add_cnots(
        self,
        circuit: stim.Circuit,
        offsets: Iterable[complex],
        ctrl: complex,
        tgt: complex,
    ) -> None:
        candidate_pairs = [(o + ctrl, o + tgt) for o in offsets]
        filtered_pairs = [
            pair
            for pair in candidate_pairs
            if pair[0] in self.all_sites and pair[1] in self.all_sites
        ]
        circuit.append(
            "CNOT", [self.site2ind[q] for pair in filtered_pairs for q in pair]
        )

    def _add_detectors(
        self, circuit: stim.Circuit, *, which_round: Literal["first", "middle", "last"]
    ) -> None:
        if which_round == "first":
            if self.basis == "X":  # X-basis memory circuit
                # Record X-type detectors only
                for k, i in enumerate(self.xmq_inds):
                    site = self.ind2site[i]
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-self.num_stabs + k)],
                        (site.real, site.imag, 0),
                    )
            else:  # Z-basis memory circuit
                # Record Z-type detectors only
                for k, i in enumerate(self.zmq_inds):
                    site = self.ind2site[i]
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-self.num_zstabs + k)],
                        (site.real, site.imag, 0),
                    )

        elif which_round == "middle":
            for k, i in enumerate(self.xmq_inds):  # X-type detectors
                site = self.ind2site[i]
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-2 * self.num_stabs + k),
                        stim.target_rec(-self.num_stabs + k),
                    ],
                    (site.real, site.imag, 0),
                )
            for k, i in enumerate(self.zmq_inds):  # Z-type detectors
                site = self.ind2site[i]
                lookback_indices: list[int] = []
                if (site - 2j) in self.zmq_sites:
                    lookback_indices.append(
                        -self.num_stabs
                        - self.num_zstabs
                        + self.zmq_site2relind[site - 2j]
                    )
                if site.imag == 0:
                    lookback_indices.append(
                        -self.num_stabs - self.num_zstabs + self.zmq_site2relind[site]
                    )
                if (site + 2j) in self.zmq_sites:
                    lookback_indices.append(
                        -self.num_stabs
                        - self.num_zstabs
                        + self.zmq_site2relind[site + 2j]
                    )
                lookback_indices.append(-self.num_zstabs + k)
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(lb) for lb in lookback_indices],
                    (site.real, site.imag, 0),
                )

        elif which_round == "last":
            pass  # TODO

        else:
            raise ValueError(f"Invalid argument: {which_round=}")

        circuit.append("SHIFT_COORDS", [], (0, 0, 1))
