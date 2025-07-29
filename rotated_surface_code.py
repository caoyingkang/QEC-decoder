import numpy as np
from typing import Optional


class RotatedSurfaceCode:
    def __init__(self, d: int):
        """
        Args:
            d (int): distance of the code, must be an odd integer.
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be an odd number")

        self.d = d
        self.n = d * d  # number of data qubits
        self.mx = (d * d - 1) // 2  # number of X-type stabilizers
        self.mz = (d * d - 1) // 2  # number of X-type stabilizers
        self._generate_parity_check_matrices()  # obtain self.Hx and self.Hz

    def _coord_to_dq(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to data qubit index."""
        assert row % 2 == 1 and col % 2 == 1, "only odd rows and columns are data qubits"
        rr = row // 2
        cc = col // 2
        assert 0 <= rr < self.d and 0 <= cc < self.d, "coordinates out of bounds"
        return rr * self.d + cc

    def _dq_to_coord(self, i: int) -> tuple[int, int]:
        """Convert data qubit index i to (row, col) coordinates."""
        assert 0 <= i < self.n, "index out of bounds"
        rr, cc = divmod(i, self.d)
        return 2 * rr + 1, 2 * cc + 1

    def _coord_to_xstab(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to X-type stabilizer index."""
        assert row % 2 == 0 and col % 2 == 0, "only even rows and columns are stabilizers"
        assert (row + col) % 4 == 2, "this is not an X-type stabilizer"
        rr = row // 2
        cc = (col - 2) // 4
        assert 0 <= rr <= self.d and 0 <= cc < self.d // 2, "coordinates out of bounds"
        return rr * (self.d // 2) + cc

    def _xstab_to_coord(self, i: int) -> tuple[int, int]:
        """Convert X-type stabilizer index i to (row, col) coordinates."""
        assert 0 <= i < self.mx, "index out of bounds"
        rr, cc = divmod(i, self.d // 2)
        row = 2 * rr
        col = 4 * cc + (2 if rr % 2 == 0 else 4)
        return row, col

    def _coord_to_zstab(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to Z-type stabilizer index."""
        assert row % 2 == 0 and col % 2 == 0, "only even rows and columns are stabilizers"
        assert (row + col) % 4 == 0, "this is not a Z-type stabilizer"
        rr = (row - 2) // 4
        cc = col // 2
        assert 0 <= rr < self.d // 2 and 0 <= cc <= self.d, "coordinates out of bounds"
        return rr * (self.d + 1) + cc

    def _zstab_to_coord(self, i: int) -> tuple[int, int]:
        """Convert Z-type stabilizer index i to (row, col) coordinates."""
        assert 0 <= i < self.mz, "index out of bounds"
        rr, cc = divmod(i, self.d + 1)
        row = 4 * rr + (4 if cc % 2 == 0 else 2)
        col = 2 * cc
        return row, col

    def _generate_parity_check_matrices(self):
        """
        Generate the X- and Z-type parity-check matrices self.Hx and self.Hz, dtype=int, values in {0, 1}.
        """
        # TODO: use csc sparse matrix
        self.Hx = np.zeros((self.mx, self.n), dtype=int)
        self.Hz = np.zeros((self.mz, self.n), dtype=int)

        # X-type stabilizers
        for i in range(self.mx):
            row, col = self._xstab_to_coord(i)
            data_qubits = []
            if row > 0 and col > 0:
                data_qubits.append(self._coord_to_dq(row - 1, col - 1))
            if row > 0 and col < 2 * self.d:
                data_qubits.append(self._coord_to_dq(row - 1, col + 1))
            if row < 2 * self.d and col > 0:
                data_qubits.append(self._coord_to_dq(row + 1, col - 1))
            if row < 2 * self.d and col < 2 * self.d:
                data_qubits.append(self._coord_to_dq(row + 1, col + 1))
            self.Hx[i, data_qubits] = 1

        # Z-type stabilizers
        for i in range(self.mz):
            row, col = self._zstab_to_coord(i)
            data_qubits = []
            if row > 0 and col > 0:
                data_qubits.append(self._coord_to_dq(row - 1, col - 1))
            if row > 0 and col < 2 * self.d:
                data_qubits.append(self._coord_to_dq(row - 1, col + 1))
            if row < 2 * self.d and col > 0:
                data_qubits.append(self._coord_to_dq(row + 1, col - 1))
            if row < 2 * self.d and col < 2 * self.d:
                data_qubits.append(self._coord_to_dq(row + 1, col + 1))
            self.Hz[i, data_qubits] = 1

    def sample_error_and_syndrome_with_code_capacity_model(self, N: int, p: np.ndarray, seed: Optional[int] = None):
        """
        Sample error patterns and their syndromes using the code capacity model (i.e., independent depolarizing 
        channel on each data qubit, error-free stabilizer measurements).

        Args:
            N (int): number of samples.
            p (np.ndarray): depolarizing error probabilities for each data qubit, shape (n,), dtype=float, values in (0, 0.5).
            seed (int): random seed for reproducibility.

        Returns:
            tuple: (ex, ez, sx, sz)
                - ex: ndarray, shape (N, n), dtype=int, X-component of the sampled errors.
                - ez: ndarray, shape (N, n), dtype=int, Z-component of the sampled errors.
                - sx: ndarray, shape (N, Hx.shape[0]), dtype=int, X-stabilizer syndromes.
                - sz: ndarray, shape (N, Hz.shape[0]), dtype=int, Z-stabilizer syndromes.
        """
        assert isinstance(p, np.ndarray) and p.shape == (self.n,)
        assert p.dtype == float and 0 < p.min() and p.max() < 0.5

        if seed is not None:
            np.random.seed(seed)

        # sample faulty data qubits
        faulty_qubits = np.random.rand(N, self.n) < p
        # sample Pauli error terms
        paulis = np.random.choice(['X', 'Y', 'Z'], size=(N, self.n))
        paulis[~faulty_qubits] = 'I'  # no error on non-faulty qubits
        # extract X and Z components of the errors
        x_mask: np.ndarray = (paulis == 'X')
        y_mask: np.ndarray = (paulis == 'Y')
        z_mask: np.ndarray = (paulis == 'Z')
        ex = (x_mask | y_mask).astype(int)
        ez = (z_mask | y_mask).astype(int)
        # compute syndromes
        sx = (ez @ self.Hx.T) % 2
        sz = (ex @ self.Hz.T) % 2

        assert ex.shape == (N, self.n) and ex.dtype == int
        assert np.all(np.isin(ex, [0, 1]))
        assert ez.shape == (N, self.n) and ez.dtype == int
        assert np.all(np.isin(ez, [0, 1]))
        assert sx.shape == (N, self.Hx.shape[0]) and sx.dtype == int
        assert np.all(np.isin(sx, [0, 1]))
        assert sz.shape == (N, self.Hz.shape[0]) and sz.dtype == int
        assert np.all(np.isin(sz, [0, 1]))
        return ex, ez, sx, sz
