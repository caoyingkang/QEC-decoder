import numpy as np
from typing import Optional


class RotatedSurfaceCode:
    def __init__(self, d):
        """
        Args:
            d (int): distance of the code (odd integer)
        """
        assert d % 2 == 1, "Distance d must be an odd number"

        self.d = d
        self.n = d * d  # number of data qubits
        self._generate_parity_check_matrices()  # obtain self.Hx and self.Hz

    def _coord_to_index(self, row, col):
        """Convert (row, col) coordinates to data qubit index i."""
        assert 0 <= row < self.d and 0 <= col < self.d
        return row * self.d + col

    def _index_to_coord(self, i):
        """Convert data qubit index i to (row, col) coordinates."""
        assert 0 <= i < self.n
        return divmod(i, self.d)

    def _generate_parity_check_matrices(self):
        """
        Generate the X- and Z-type parity-check matrices self.Hx and self.Hz, dtype=int, values in {0, 1}.
        """
        x_checks = []
        z_checks = []
        # stabilizer generators in the bulk of the lattice
        for row in range(self.d - 1):
            for col in range(self.d - 1):
                check = np.zeros(self.n, dtype=int)
                check[self._coord_to_index(row, col)] = 1
                check[self._coord_to_index(row, col + 1)] = 1
                check[self._coord_to_index(row + 1, col)] = 1
                check[self._coord_to_index(row + 1, col + 1)] = 1
                if (row + col) % 2 == 0:
                    x_checks.append(check)
                else:
                    z_checks.append(check)
        # stabilizer generators on the upper boundary
        for col in range(0, self.d - 1, 2):
            check = np.zeros(self.n, dtype=int)
            check[self._coord_to_index(0, col)] = 1
            check[self._coord_to_index(0, col + 1)] = 1
            z_checks.append(check)
        # stabilizer generators on the lower boundary
        for col in range(1, self.d - 1, 2):
            check = np.zeros(self.n, dtype=int)
            check[self._coord_to_index(self.d - 1, col)] = 1
            check[self._coord_to_index(self.d - 1, col + 1)] = 1
            z_checks.append(check)
        # stabilizer generators on the left boundary
        for row in range(1, self.d - 1, 2):
            check = np.zeros(self.n, dtype=int)
            check[self._coord_to_index(row, 0)] = 1
            check[self._coord_to_index(row + 1, 0)] = 1
            x_checks.append(check)
        # stabilizer generators on the right boundary
        for row in range(0, self.d - 1, 2):
            check = np.zeros(self.n, dtype=int)
            check[self._coord_to_index(row, self.d - 1)] = 1
            check[self._coord_to_index(row + 1, self.d - 1)] = 1
            x_checks.append(check)
        # convert to numpy arrays
        self.Hx = np.array(x_checks, dtype=int)
        self.Hz = np.array(z_checks, dtype=int)

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
