from typing import Literal
from typing_extensions import Self

import ldpc.mod2
import numpy as np
import stim

from ..types import Bit2DArray, Bool4DArray, Int2DArray
from .base import QECCircuit

# Depth-7 syndrome-measurement cycle from Bravyi et al. (arXiv:2308.07915):
# entry t gives the Tanner-graph direction each check couples to at CNOT round t
# (None = the check idles). Directions 0-2 address left data qubits, 3-5 right
# data qubits.
_SX = (None, 1, 4, 3, 5, 0, 2)
_SZ = (3, 5, 0, 1, 2, 4, None)


class BBCode_Circuit(QECCircuit, registry_name="BBCode_Circuit"):
    """Memory circuit for a bivariate bicycle (BB) code, generated from its
    algebraic definition (Bravyi et al., arXiv:2308.07915).

    The code lives on a ``Z_ell x Z_m`` torus and is defined by two polynomials
    ``A`` and ``B``, each a sum of three monomials ``x^i y^j`` (``x``/``y`` are
    the torus shift operators), with check matrices ``hx = [A|B]`` and
    ``hz = [B^T|A^T]``. The syndrome-measurement cycle is the depth-7 CNOT
    schedule from the same paper, with uniform circuit-level noise. For the
    gross code [[144,12,12]] (``ell=12, m=6, A = x^3+y+y^2, B = y^3+x+x^2``)
    the generated circuit reproduces the pre-generated ``.stim`` files used by
    ``BB_144_12_12_Circuit``, up to the choice of logical-observable basis
    (see ``tests/test_bb_code_circuit.py``).

    Qubit layout: left data ``[0, n2)``, right data ``[n2, 2*n2)``, X-checks
    ``[2*n2, 3*n2)``, Z-checks ``[3*n2, 4*n2)``, where ``n2 = ell * m`` and
    torus site ``(i, j)`` has index ``i * m + j``.

    Besides the ``stim.Circuit``, this class exposes the torus geometry used by
    CNN decoders: each in-basis check (and its detector per time slice) sits on
    a dense ``(ell, m)`` grid, data qubits occupy two such planes (left/right),
    and the check<->data couplings are translation-invariant offsets.
    """

    def __init__(
        self,
        *,
        ell: int,
        m: int,
        a_poly: list[tuple[int, int]],
        b_poly: list[tuple[int, int]],
        basis: Literal["X", "Z"],
        rounds: int,
        error_rate: float,
        filter_detectors: bool = True,
    ):
        """
        Parameters
        ----------
            ell, m : int
                Torus dimensions; the code has ``2 * ell * m`` data qubits.
            a_poly, b_poly : list[tuple[int, int]]
                The three monomials ``(x_power, y_power)`` of the polynomials
                ``A`` and ``B``. The gross code [[144,12,12]] is
                ``a_poly=[(3,0),(0,1),(0,2)]``, ``b_poly=[(0,3),(1,0),(2,0)]``.
            basis : Literal['X', 'Z']
                Basis of logical state preparation and measurement. If
                basis='X' (resp. 'Z'), then we will use X-type (resp. Z-type)
                stabilizer measurement outcomes to correct Pauli Z (resp. X)
                errors.
            rounds : int
                Number of syndrome measurement cycles. Must be at least 2.
            error_rate : float
                Uniform error rate of all fault locations.
            filter_detectors : bool
                If True (default), only in-basis detectors are emitted. If
                False, off-basis detectors are also emitted (matching the
                unfiltered pre-generated ``.stim`` files).
        """
        if ell < 1 or m < 1:
            raise ValueError("ell and m must be positive")
        if rounds < 2:
            raise ValueError("rounds must be at least 2")
        if basis not in ["X", "Z"]:
            raise ValueError("Basis must be 'X' or 'Z'")
        a_poly = [(dx % ell, dy % m) for dx, dy in a_poly]
        b_poly = [(dx % ell, dy % m) for dx, dy in b_poly]
        if len(a_poly) != 3 or len(set(a_poly)) != 3:
            raise ValueError("a_poly must consist of 3 distinct monomials")
        if len(b_poly) != 3 or len(set(b_poly)) != 3:
            raise ValueError("b_poly must consist of 3 distinct monomials")

        self.ell = ell
        self.m = m
        self.a_poly = a_poly
        self.b_poly = b_poly
        self.basis = basis
        self.rounds = rounds
        self.error_rate = error_rate
        self.filter_detectors = filter_detectors

        self.n2 = n2 = ell * m  # checks of each type = data qubits per plane

        # Check -> data neighbor tables, (n2, 3), one column per monomial.
        # X-check i couples left data j iff A[i, j] = 1, i.e. j = i + offset;
        # Z-check i couples left data j iff B^T[i, j] = B[j, i] = 1, i.e.
        # j = i - offset (offsets act on torus sites, with wraparound).
        sites = np.arange(n2, dtype=np.int64)
        il, im = sites // m, sites % m

        def shifted(dx: int, dy: int) -> np.ndarray:
            return ((il + dx) % ell) * m + (im + dy) % m

        self._xck_dataL = np.stack([shifted(dx, dy) for dx, dy in a_poly], axis=1)
        self._xck_dataR = np.stack([shifted(dx, dy) for dx, dy in b_poly], axis=1)
        self._zck_dataL = np.stack([shifted(-dx, -dy) for dx, dy in b_poly], axis=1)
        self._zck_dataR = np.stack([shifted(-dx, -dy) for dx, dy in a_poly], axis=1)

        self.hx = self._check_matrix(self._xck_dataL, self._xck_dataR)
        self.hz = self._check_matrix(self._zck_dataL, self._zck_dataR)
        self._observable_data_supports = self._compute_observable_data_supports()

        super().__init__(self._build_circuit())

    @classmethod
    def with_uniform_error_rate(
        cls,
        error_rate: float,
        *,
        ell: int,
        m: int,
        a_poly: list[tuple[int, int]],
        b_poly: list[tuple[int, int]],
        basis: Literal["X", "Z"],
        rounds: int,
        filter_detectors: bool = True,
    ) -> Self:
        return cls(
            ell=ell,
            m=m,
            a_poly=a_poly,
            b_poly=b_poly,
            basis=basis,
            rounds=rounds,
            error_rate=error_rate,
            filter_detectors=filter_detectors,
        )

    def _check_matrix(self, nbrs_left: Int2DArray, nbrs_right: Int2DArray) -> Bit2DArray:
        """Stabilizer check matrix, shape=(n2, 2*n2), from neighbor tables."""
        h = np.zeros((self.n2, 2 * self.n2), dtype=np.uint8)
        rows = np.arange(self.n2)[:, None]
        h[rows, nbrs_left] = 1
        h[rows, self.n2 + nbrs_right] = 1
        return h

    # ----------------------------------------------------------------------------------------
    # Logical observables
    # ----------------------------------------------------------------------------------------

    @property
    def observable_data_supports(self) -> Bit2DArray:
        """
        Data-qubit supports of the logical observables, shape=(k, 2*n2).
        Basis-'Z' observables are logical Z operators (kernel of ``hx`` modulo
        the row space of ``hz``); basis-'X' the transpose case. The basis
        choice is the deterministic pivoting of ``ldpc.mod2``; it spans the
        same logical group as, but does not coincide with, the choice baked
        into the pre-generated ``BB_144_12_12_Circuit`` files.
        """
        return self._observable_data_supports

    def _compute_observable_data_supports(self) -> Bit2DArray:
        h_same, h_other = (
            (self.hz, self.hx) if self.basis == "Z" else (self.hx, self.hz)
        )
        kernel = ldpc.mod2.nullspace(h_other).toarray().astype(np.uint8)
        stack = np.vstack([h_same, kernel])
        pivots = ldpc.mod2.pivot_rows(stack)
        logical_rows = [int(r) for r in pivots if r >= h_same.shape[0]]
        return stack[logical_rows]

    # ----------------------------------------------------------------------------------------
    # CNN-grid geometry
    # ----------------------------------------------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """
        Shape (num_layers, ell, m) of the dense spacetime grid of in-basis
        detectors: ``rounds`` measurement cycles + 1 final-measurement slice,
        with one detector per torus site per layer.
        """
        return self.rounds + 1, self.ell, self.m

    @property
    def detector_grid_sites(self) -> Int2DArray:
        """
        Grid site of each detector, shape=(num_detectors, 3). Each row is
        (layer, row, col) with row in [0, ell) and col in [0, m). Detector
        index = layer * n2 + i, with i the in-basis check index. Only
        available when ``filter_detectors`` is True (otherwise off-basis
        detectors share the grid).
        """
        if not self.filter_detectors:
            raise RuntimeError(
                "detector_grid_sites requires filter_detectors=True"
            )
        num_layers, _, _ = self.grid_shape
        sites = np.arange(self.n2, dtype=np.int64)
        per_layer = np.stack([sites // self.m, sites % self.m], axis=1)
        return np.hstack(
            [
                np.repeat(np.arange(num_layers, dtype=np.int64), self.n2)[:, None],
                np.tile(per_layer, (num_layers, 1)),
            ]
        )

    @property
    def check_dataL_offsets(self) -> Int2DArray:
        """Torus offsets (drow, dcol) from an in-basis check site to its three
        left-plane data neighbors, shape=(3, 2)."""
        poly = self.a_poly if self.basis == "X" else self.b_poly
        sign = 1 if self.basis == "X" else -1
        return np.array(
            [((sign * dx) % self.ell, (sign * dy) % self.m) for dx, dy in poly],
            dtype=np.int64,
        )

    @property
    def check_dataR_offsets(self) -> Int2DArray:
        """Torus offsets (drow, dcol) from an in-basis check site to its three
        right-plane data neighbors, shape=(3, 2)."""
        poly = self.b_poly if self.basis == "X" else self.a_poly
        sign = 1 if self.basis == "X" else -1
        return np.array(
            [((sign * dx) % self.ell, (sign * dy) % self.m) for dx, dy in poly],
            dtype=np.int64,
        )

    @property
    def check_data_nbrs(self) -> Int2DArray:
        """
        Data-qubit neighbors of each in-basis check, shape=(n2, 6). Columns
        0-2 are left-plane neighbors (data ids in [0, n2)), columns 3-5
        right-plane neighbors (data ids in [n2, 2*n2)).
        """
        if self.basis == "X":
            left, right = self._xck_dataL, self._xck_dataR
        else:
            left, right = self._zck_dataL, self._zck_dataR
        return np.hstack([left, self.n2 + right])

    @property
    def observable_grid_masks(self) -> Bool4DArray:
        """
        Per-observable data-qubit support masks, shape=(num_observables, 2,
        ell, m). The second axis indexes the left/right data plane.
        """
        k = self._observable_data_supports.shape[0]
        return (
            self._observable_data_supports.reshape(k, 2, self.ell, self.m)
            .astype(np.bool_)
        )

    # ----------------------------------------------------------------------------------------
    # Circuit construction
    # ----------------------------------------------------------------------------------------

    def _build_circuit(self) -> stim.Circuit:
        p = self.error_rate
        n2, rounds = self.n2, self.rounds
        data_l = list(range(n2))
        data_r = list(range(n2, 2 * n2))
        xq = list(range(2 * n2, 3 * n2))
        zq = list(range(3 * n2, 4 * n2))
        data_meas_gate = "M" if self.basis == "Z" else "MX"

        circ = stim.Circuit()

        def reset_data(qubits: list[int]) -> None:
            if self.basis == "Z":
                circ.append("R", qubits)
                circ.append("X_ERROR", qubits, p)
            else:
                circ.append("RX", qubits)
                circ.append("Z_ERROR", qubits, p)

        def cx_block(controls: np.ndarray, targets: np.ndarray) -> None:
            order = np.argsort(controls)
            flat = np.stack([controls[order], targets[order]], axis=1).reshape(-1)
            circ.append("CX", flat.tolist())
            circ.append("DEPOLARIZE2", flat.tolist(), p)

        check_inds = np.arange(n2, dtype=np.int64)

        def x_block(t: int) -> None:
            d = _SX[t]
            targets = (
                self._xck_dataL[:, d] if d < 3 else n2 + self._xck_dataR[:, d - 3]
            )
            cx_block(2 * n2 + check_inds, targets)

        def z_block(t: int) -> None:
            d = _SZ[t]
            controls = (
                self._zck_dataL[:, d] if d < 3 else n2 + self._zck_dataR[:, d - 3]
            )
            cx_block(controls, 3 * n2 + check_inds)

        # Measurement bookkeeping: absolute measurement index per outcome.
        meas_count = 0
        z_meas = np.zeros((rounds, n2), dtype=np.int64)
        x_meas = np.zeros((rounds, n2), dtype=np.int64)
        data_meas = np.zeros(2 * n2, dtype=np.int64)

        # Initial state preparation (the X-check preparation belongs to the
        # first cycle below).
        reset_data(data_r)
        circ.append("R", zq)
        circ.append("X_ERROR", zq, p)
        reset_data(data_l)

        for k in range(rounds):
            last = k == rounds - 1
            # CNOT round 0: prep X-checks, couple Z-checks. The data qubits
            # idling here get their depolarizing noise emitted just before the
            # preparation (and none in the very first cycle, matching the
            # pre-generated circuits).
            if k > 0:
                circ.append("DEPOLARIZE1", data_l, p)
            circ.append("RX", xq)
            circ.append("Z_ERROR", xq, p)
            z_block(0)
            # CNOT rounds 1-5: couple both check types each round.
            for t in range(1, 5):
                x_block(t)
                z_block(t)
            if last:
                # In the final cycle the pre-generated circuits emit a single
                # idle-noise layer on the right data and Z-check qubits after
                # CNOT round 4, instead of the per-round idles below.
                circ.append("DEPOLARIZE1", data_r + zq, p)
                x_block(5)
                z_block(5)
                x_block(6)
            else:
                x_block(5)
                z_block(5)
                # CNOT round 6: only X-checks couple; right data qubits idle.
                circ.append("DEPOLARIZE1", data_r, p)
                x_block(6)
            # Measure Z-checks, then X-checks (data qubits idle in between).
            circ.append("M", zq, p)
            z_meas[k] = meas_count + check_inds
            meas_count += n2
            if not last:
                circ.append("DEPOLARIZE1", data_l + data_r, p)
            circ.append("MX", xq, p)
            x_meas[k] = meas_count + check_inds
            meas_count += n2
            if last:
                # Final transversal data measurement (noiseless readout),
                # interleaved left/right.
                interleaved = np.stack(
                    [check_inds, n2 + check_inds], axis=1
                ).reshape(-1)
                circ.append(data_meas_gate, interleaved.tolist())
                data_meas[interleaved] = meas_count + np.arange(2 * n2)
                meas_count += 2 * n2
            else:
                circ.append("R", zq)
                circ.append("X_ERROR", zq, p)

        # Detectors. In-basis checks are deterministic in the first cycle;
        # afterwards consecutive cycles are compared; the final slice compares
        # the last cycle against the data measurement. Off-basis checks only
        # support cycle-to-cycle comparisons.
        in_meas = z_meas if self.basis == "Z" else x_meas
        in_nbrs = self.check_data_nbrs

        def rec(meas_index: int) -> stim.GateTarget:
            return stim.target_rec(meas_index - meas_count)

        def coords(i: int, layer: int) -> list[float]:
            return [i % self.m, i // self.m, layer]

        for i in range(n2):
            circ.append("DETECTOR", [rec(in_meas[0, i])], coords(i, 0))
        for k in range(1, rounds):
            pair_meas = (z_meas, x_meas) if not self.filter_detectors else (in_meas,)
            for meas in pair_meas:
                for i in range(n2):
                    circ.append(
                        "DETECTOR",
                        [rec(meas[k - 1, i]), rec(meas[k, i])],
                        coords(i, k),
                    )
        for i in range(n2):
            targets = [rec(in_meas[rounds - 1, i])]
            targets += [rec(data_meas[q]) for q in in_nbrs[i]]
            circ.append("DETECTOR", targets, coords(i, rounds))

        # Logical observables, read off the final data measurement.
        for j, support in enumerate(self._observable_data_supports):
            targets = [rec(data_meas[q]) for q in np.nonzero(support)[0]]
            circ.append("OBSERVABLE_INCLUDE", targets, j)

        return circ
