import numpy as np
from typing import Optional, Tuple


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
        self.k = 1  # number of logical qubits
        self._construct_stabilizer_matrices()  # obtain self.Hx and self.Hz
        self._construct_logical_operator_matrices()  # obtain self.Lx and self.Lz

        assert np.all((self.Hx @ self.Hz.T) % 2 == 0)
        assert np.all((self.Hx @ self.Lz.T) % 2 == 0)
        assert np.all((self.Hz @ self.Lx.T) % 2 == 0)
        assert np.all((self.Lx @ self.Lz.T) % 2 == np.eye(self.k, dtype=int))

    def get_check_matrices_and_action_matrices(self, noise_model: str, num_round: Optional[int] = None
                                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the check matrices and action matrices for the specified noise model.

        Parameters
        ----------
            noise_model : str
                'code-capacity' or 'phenomenological'.
            
            num_round : int or None
                Number of rounds of stabilizer measurement. If None, defaults to the code distance. This parameter is ignored 
                when noise_model is 'code-capacity'.

        Returns
        -------
            cmx : ndarray
                X-type check matrix, shape=(# of X-type detectors, # of Z-type error mechanisms), dtype=int, values in {0, 1}.
                A nonzero entry at row i and column j indicates that the i-th detector is flipped by the j-th error mechanism.
            
            amx : ndarray
                X-type action matrix, shape=(# of logical qubits, # of Z-type error mechanisms), dtype=int, values in {0, 1}.
                A nonzero entry at row i and column j indicates that the eigenvalue of the logical X_i operator is flipped by 
                the j-th error mechanism.

            cmz : ndarray
                Z-type check matrix, shape=(# of Z-type detectors, # of X-type error mechanisms), dtype=int, values in {0, 1}.
                A nonzero entry at row i and column j indicates that the i-th detector is flipped by the j-th error mechanism.
            
            amz : ndarray
                Z-type action matrix, shape=(# of logical qubits, # of X-type error mechanisms), dtype=int, values in {0, 1}.
                A nonzero entry at row i and column j indicates that the eigenvalue of the logical Z_i operator is flipped by 
                the j-th error mechanism.

        Notes
        -----
            'code-capacity' noise model:
                For the X-type (similarly for Z-type) decoding problem, this noise model assumes a single round of perfect 
                stabilizer measurement and only considers Pauli Z errors on data qubits; the detectors are simply the 
                measurement outcome of X-type stabilizers. Hence cmx is nothing but the X-type stabilizer matrix, and amx
                is nothing but the matrix representation of logical X operators.
            
            'phenomenological' noise model:
                For the X-type (similarly for Z-type) decoding problem, this noise model assumes multiple rounds of noisy 
                stabilizer measurement, considering in each round both Pauli Z errors on data qubits and bit-flip errors 
                on the X-type stabilizer measurement outcomes; the detectors are the *change* between two measured outcomes 
                of the same X-type stabilizer in consecutive rounds. More precisely, X-type detectors are defined as follows:

                    - D_{0,i} = meas. outcome of X-type stabilizer i in round 0,

                    - D_{t,i} = XOR of the two meas. outcomes of X-type stabilizer i in round t-1 and round t, for 1 <= t < #round.
                
                Z-type error mechanisms are defined as follows:

                    - E1_{t,j} = Pauli Z error on data qubit j happening just before round t, for 0 <= t < #round.

                    - E2_{t,i} = bit-flip error on the meas. outcome of X-type stabilizer i in round t, for 0 <= t < #round - 1. Note 
                    that by convention, we assume that the last round of stabilizer measurement is error-free.
                
                The matrix cmx can be written as two parts as cmx = [cmx1, cmx2], where cmx1 consists of the first #round * #data_qubit 
                columns of cmx, and cmx2 consists of the last (#round - 1) * #x_stabilizer columns of cmx. Detector D_{t,i} corresponds to 
                row (t * #x_stabilizer + i) in cmx. Error mechanism E1_{t,j} corresponds to column (t * #data_qubit + j) in cmx1, and 
                error mechanism E2_{t,i} corresponds to column (t * #x_stabilizer + i) in cmx2.

                The matrix amx can be written as two parts as amx = [amx1, amx2] in the same way as above, and the second part amx2 is 
                the all-zeroes matrix.
        """
        if num_round is None:
            num_round = self.d

        if noise_model == 'code-capacity':
            cmx = np.copy(self.Hx)
            cmz = np.copy(self.Hz)
            amx = np.copy(self.Lx)
            amz = np.copy(self.Lz)
        elif noise_model == 'phenomenological':
            cmx = self._cm_phenomenological('X', num_round)
            cmz = self._cm_phenomenological('Z', num_round)
            amx = self._am_phenomenological('X', num_round)
            amz = self._am_phenomenological('Z', num_round)
        else:
            # TODO: implement 'circuit-level' noise model
            raise ValueError("unknown error model")

        return cmx, amx, cmz, amz

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

    def _construct_stabilizer_matrices(self):
        """
        Construct the X- and Z-type stabilizer matrices self.Hx and self.Hz, ndim=2, dtype=int, values in {0, 1}.
        The i-th X-type (similarly for Z-type) stabilizer is the tensor product of Pauli X operators acting 
        on the data qubits indexed by the the i-th row of self.Hx.
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

    def _construct_logical_operator_matrices(self):
        """Construct the matrices self.Lx and self.Lz representing logical X and Z operators, ndim=2, dtype=int, values in {0, 1}.
        The i-th logical X (similarly for Z) operator is the tensor product of Pauli X operators acting on the data qubits indexed 
        by the the i-th row of self.Hx.
        """
        # TODO: use csc sparse matrix
        self.Lx = np.zeros((1, self.n), dtype=int)
        self.Lz = np.zeros((1, self.n), dtype=int)

        # logical X
        data_qubits = [self._coord_to_dq(row, 1)
                       for row in range(1, 2 * self.d, 2)]
        self.Lx[0, data_qubits] = 1

        # logical Z
        data_qubits = [self._coord_to_dq(1, col)
                       for col in range(1, 2 * self.d, 2)]
        self.Lz[0, data_qubits] = 1


    def _cm_phenomenological(self, detector_type: str, num_round: int) -> np.ndarray:
        """
        Generate the check matrix of the specified detector type for the phenomenological noise model.
        """
        n = self.n
        m = self.mx if detector_type == 'X' else self.mz
        H = self.Hx if detector_type == 'X' else self.Hz

        num_detectors = num_round * m
        num_dq_errors = num_round * n
        num_meas_errors = (num_round - 1) * m
        num_errors = num_dq_errors + num_meas_errors

        cm1 = np.zeros((num_detectors, num_dq_errors), dtype=int)
        for t in range(num_round):
            cm1[t * m:(t + 1) * m, t * n:(t + 1) * n] = H.copy()

        cm2 = np.zeros((num_detectors, num_meas_errors), dtype=int)
        for t in range(num_round - 1):
            cm2[t * m:(t + 1) * m, t * m:(t + 1) * m] = np.eye(m, dtype=int)
        for t in range(1, num_round):
            cm2[t * m:(t + 1) * m, (t - 1) * m:t * m] = np.eye(m, dtype=int)

        cm = np.hstack((cm1, cm2))

        assert cm.shape == (num_detectors, num_errors)
        assert cm.dtype == int
        assert np.all(np.isin(cm, [0, 1]))
        return cm

    def _am_phenomenological(self, logical_type: str, num_round: int) -> np.ndarray:
        """
        Generate the action matrix of the specified logical type for the phenomenological noise model.
        """
        n = self.n
        m = self.mx if logical_type == 'X' else self.mz
        L = self.Lx if logical_type == 'X' else self.Lz

        num_dq_errors = num_round * n
        num_meas_errors = (num_round - 1) * m
        num_errors = num_dq_errors + num_meas_errors

        am1 = np.hstack([L.copy()] * num_round)
        am2 = np.zeros((1, num_meas_errors), dtype=int)

        am = np.hstack((am1, am2))

        assert am.shape == (1, num_errors)
        assert am.dtype == int
        assert np.all(np.isin(am, [0, 1]))
        return am
