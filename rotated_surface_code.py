import numpy as np
from typing import Optional


class RotatedSurfaceCode:
    def __init__(self, d: int):
        """
        Parameters
        ----------
            d : int
                Distance of the code, must be an odd integer at least 3.
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be an odd number")
        if d < 3:
            raise ValueError("Distance d must be at least 3")

        self.d = d
        self.n = d * d  # number of data qubits
        self.mx = (d * d - 1) // 2  # number of X-type stabilizers
        self.mz = (d * d - 1) // 2  # number of X-type stabilizers
        self._calculate_stabilizer_matrices()  # obtain self.Hx and self.Hz

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

    def _calculate_stabilizer_matrices(self):
        """
        Generate the X- and Z-type stabilizer matrices self.Hx and self.Hz, dtype=int, values in {0, 1}.
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

    def get_parity_check_matrix(self, detector_type: str, noise_model: str, num_round: Optional[int] = None) -> np.ndarray:
        """
        Get the parity check matrix for the specified detector type and noise model.

        Parameters
        ----------
            detector_type : str
                'X' or 'Z'.

            noise_model : str
                'code-capacity' or 'phenomenological'.
            
            num_round : int or None
                Number of rounds of stabilizer measurement. If None, defaults to the code distance. This parameter is ignored 
                when noise_model is 'code-capacity'.

        Returns
        -------
            pcm : ndarray
                Parity check matrix, dtype=int, values in {0, 1}. A nonzero entry at row i and column j indicates that the 
                i-th detector is flipped by the j-th error mechanism.

        Notes
        -----
            'code-capacity' noise model:
                When detector_type is 'X' (similarly for 'Z'), this noise model assumes a single round of perfect stabilizer 
                measurement and only considers Pauli Z errors on data qubits; the detectors are simply the meas. outcome of 
                X-type stabilizers. Hence the pcm is nothing but the X-type stabilizer matrix.
            
            'phenomenological' noise model:
                When detector_type is 'X' (similarly for 'Z'), this noise model assumes multiple rounds of noisy stabilizer 
                measurement, considering in each round both Pauli Z errors on data qubits and bit-flip errors on the X-type 
                stabilizer meas. outcomes; the detectors are the *change* between two meas. outcomes of the same X-type 
                stabilizer in consecutive rounds. More precisely, the detectors are defined as follows:

                    - D_{0,i} = meas. outcome of X-type stabilizer i in round 0,

                    - D_{t,i} = XOR of the two meas. outcomes of X-type stabilizer i in round t-1 and round t, for 1 <= t < #round.
                
                The error mechanisms are defined as follows:

                    - E1_{t,j} = Pauli Z error on data qubit j happening just before round t, for 0 <= t < #round.

                    - E2_{t,i} = bit-flip error on the meas. outcome of X-type stabilizer i in round t, for 0 <= t < #round.
                
                The pcm has shape (#detector, #error_mechanism) and can be written as two parts as pcm = [pcm1, pcm2], where pcm1 
                consists of the first (#round * #data_qubit) columns, and pcm2 consists of the last (#round * #x_stabilizer) columns.
                The detector D_{t,i} corresponds to row (t * #x_stabilizer + i) in pcm, the error mechanism E1_{t,j} corresponds to 
                column (t * #data_qubit + j) in pcm1, and the error mechanism E2_{t,i} corresponds to column (t * #x_stabilizer + i) 
                in pcm2.
        """
        if detector_type not in ['X', 'Z']:
            raise ValueError("detector type must be 'X' or 'Z'")

        if num_round is None:
            num_round = self.d

        if noise_model == 'code-capacity':
            pcm = np.copy(self.Hx if detector_type == 'X' else self.Hz)
        elif noise_model == 'phenomenological':
            pcm = self._pcm_phenomelogical(detector_type, num_round)
        else:
            # TODO: implement 'circuit-level' noise model
            raise ValueError("unknown error model")

        return pcm

    def _pcm_phenomelogical(self, detector_type: str, num_round: int) -> np.ndarray:
        """
        Generate the parity check matrix for the phenomenological noise model.
        """
        if detector_type == 'X':
            num_detectors = num_round * self.mx
            num_dq_errors = num_round * self.n
            num_meas_errors = num_round * self.mx
            num_errors = num_dq_errors + num_meas_errors

            pcm1 = np.zeros((num_detectors, num_dq_errors), dtype=int)
            for t in range(num_round):
                pcm1[t * self.mx:(t + 1) * self.mx,
                     t * self.n:(t + 1) * self.n] = self.Hx

            pcm2 = np.zeros((num_detectors, num_meas_errors), dtype=int)
            for t in range(num_round):
                pcm2[t * self.mx:(t + 1) * self.mx,
                     t * self.mx:(t + 1) * self.mx] = np.eye(self.mx, dtype=int)
                if t > 0:
                    pcm2[t * self.mx:(t + 1) * self.mx,
                         (t - 1) * self.mx:t * self.mx] = np.eye(self.mx, dtype=int)

            pcm = np.hstack((pcm1, pcm2))

        else:  # detector_type == 'Z'
            num_detectors = num_round * self.mz
            num_dq_errors = num_round * self.n
            num_meas_errors = num_round * self.mz
            num_errors = num_dq_errors + num_meas_errors

            pcm1 = np.zeros((num_detectors, num_dq_errors), dtype=int)
            for t in range(num_round):
                pcm1[t * self.mz:(t + 1) * self.mz,
                     t * self.n:(t + 1) * self.n] = self.Hz

            pcm2 = np.zeros((num_detectors, num_meas_errors), dtype=int)
            for t in range(num_round):
                pcm2[t * self.mz:(t + 1) * self.mz,
                     t * self.mz:(t + 1) * self.mz] = np.eye(self.mz, dtype=int)
                if t > 0:
                    pcm2[t * self.mz:(t + 1) * self.mz,
                         (t - 1) * self.mz:t * self.mz] = np.eye(self.mz, dtype=int)

            pcm = np.hstack((pcm1, pcm2))

        assert pcm.shape == (num_detectors, num_errors)
        assert pcm.dtype == int
        assert np.all(np.isin(pcm, [0, 1]))
        return pcm


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
