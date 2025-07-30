import numpy as np
from typing import Literal, Tuple


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

    def get_check_matrix_action_matrix_probability_vector(self,
                                                          noise_model: Literal['code-capacity', 'phenomenological'],
                                                          detector_type: Literal['X', 'Z'],
                                                          **kwargs
                                                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given the specified noise model, detector type, and other noise parameters, generate the triple (check matrix, action 
        matrix, probability vector) which are used to construct a decoder.

        Parameters
        ----------
            noise_model : Literal['code-capacity', 'phenomenological']

            detector_type : Literal['X', 'Z']

            **kwargs :
                Additional parameters for the noise model.

                num_round : int
                    Number of rounds of stabilizer measurements. This argument must be provided for the 'phenomenological' noise 
                    model. If provided, it must be at least 3.

                dq_error_rate : ndarray or float
                    Error rates of each data qubit, shape=(#data_qubits,), values in (0, 0.5). If a scalar is provided, it is 
                    assumed to be the same for all data qubits. This argument must be provided for all noise models.

                meas_error_rate : ndarray or float
                    Measurement error rates of each stabilizer (of the specified type), shape=(#stabilizers,), values in (0, 0.5). 
                    If a scalar is provided, it is assumed to be the same for all stabilizers. This argument must be provided for 
                    the 'phenomenological' noise model.

        Returns
        -------
            cm : ndarray
                Check matrix, shape=(#detectors, #error_mechanisms), values in {0, 1}.
                A nonzero entry at row i and column j indicates that the i-th detector is flipped by the j-th error mechanism.
            
            am : ndarray
                Action matrix, shape=(#logical_qubits, #error_mechanisms), values in {0, 1}.
                A nonzero entry at row i and column j indicates that the eigenvalue of the i-th logical operator (of the 
                specified type) is flipped by the j-th error mechanism.

            p : ndarray
                Probability vector, shape=(#error_mechanisms,), values in [0, 1].
                The j-th entry is the probability that the j-th error mechanism occurs.

        Notes
        -----
            'code-capacity' noise model:
                When detector_type is 'X' (similarly for 'Z'), this noise model assumes a single round of perfect 
                stabilizer measurement and only considers Pauli Z errors on data qubits. The detectors are simply the 
                measurement outcome of X-type stabilizers. Hence the check matrix and action matrix are nothing but the 
                matrix representation of X-type stabilizers and logical X operators, respectively.
            
            'phenomenological' noise model:
                When detector_type is 'X' (similarly for 'Z'), this noise model assumes multiple rounds of noisy stabilizer 
                measurements. At the beginning of each round, each data qubit suffers a Pauli Z error with probability 
                dq_error_rate; at the end of each round (except for the last round, which is by convention assumed to be 
                error-free), each X-type stabilizer measurement outcome is flipped with probability meas_error_rate. The 
                detectors are the *change* between two measured outcomes of the same stabilizer in consecutive rounds. 
                More precisely, the detectors are defined as follows:

                    - D_{0,i} = meas. outcome of X-type stabilizer i in round 0,

                    - D_{t,i} = XOR of the two meas. outcomes of X-type stabilizer i in round t-1 and round t, for 1 <= t < #round.
                
                The error mechanisms are defined as follows:

                    - E1_{t,j} = Pauli Z error on data qubit j happening at the beginning of round t, for 0 <= t < #round.

                    - E2_{t,i} = bit-flip error on the meas. outcome of X-type stabilizer i in round t, for 0 <= t < #round - 1.
                
                The matrix cm can be written as two parts as cm = [cm1, cm2], where cm1 describes how the detectors are affected by 
                error mechanisms of the first category (E1), and cm2 for the second category (E2). Detector D_{t,i} corresponds to 
                row (t * #x_stabilizers + i) in cm. Error mechanism E1_{t,j} corresponds to column (t * #data_qubits + j) in cm1, and 
                E2_{t,i} corresponds to column (t * #x_stabilizers + i) in cm2.

                The matrix am can be written as two parts as am = [am1, am2] in the same way as above. Since measurement errors do 
                not affect the logical information, the second part am2 is the all-zeroes matrix.
        """
        if detector_type not in ['X', 'Z']:
            raise ValueError("detector_type must be 'X' or 'Z'")

        # Parse dq_error_rate
        if 'dq_error_rate' not in kwargs:
            raise ValueError("dq_error_rate must be provided")
        dq_error_rate = kwargs['dq_error_rate']
        if isinstance(dq_error_rate, float):
            dq_error_rate = np.full(self.n, dq_error_rate)
        assert dq_error_rate.shape == (self.n,)
        assert np.min(dq_error_rate) >= 0 and np.max(dq_error_rate) <= 0.5

        # Parse num_round and meas_error_rate
        if noise_model == 'phenomenological':
            if 'num_round' not in kwargs:
                raise ValueError("num_round must be provided")
            num_round: int = kwargs['num_round']
            assert num_round >= 3

            if 'meas_error_rate' not in kwargs:
                raise ValueError("meas_error_rate must be provided")
            meas_error_rate = kwargs['meas_error_rate']
            m = self.mx if detector_type == 'X' else self.mz
            if isinstance(meas_error_rate, float):
                meas_error_rate = np.full(m, meas_error_rate)
            assert meas_error_rate.shape == (m,)
            assert np.min(meas_error_rate) >= 0 and np.max(
                meas_error_rate) <= 0.5

        # Construct cm, am, p
        if noise_model == 'code-capacity':
            if detector_type == 'X':
                cm = self.Hx
                am = self.Lx
            else:
                cm = self.Hz
                am = self.Lz
            p = dq_error_rate
        elif noise_model == 'phenomenological':
            cm, am, p = self._cm_am_p_phenomenological(
                detector_type, num_round, dq_error_rate, meas_error_rate)
        else:
            # TODO: implement 'circuit-level' noise model
            raise ValueError("unknown error model")

        return cm, am, p

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

    def _cm_am_p_phenomenological(self,
                                  detector_type: Literal['X', 'Z'],
                                  num_round: int,
                                  dq_error_rate: np.ndarray,
                                  meas_error_rate: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the check matrix, action matrix, and probability vector for the phenomenological noise model.
        """
        n = self.n
        m = self.mx if detector_type == 'X' else self.mz
        H = self.Hx if detector_type == 'X' else self.Hz
        L = self.Lx if detector_type == 'X' else self.Lz

        num_detectors = num_round * m
        num_dq_errors = num_round * n
        num_meas_errors = (num_round - 1) * m
        num_errors = num_dq_errors + num_meas_errors

        # Construct cm
        # TODO: clever way to construct cm1 and cm2
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

        # Construct am
        am1 = np.hstack([L] * num_round)
        am2 = np.zeros((1, num_meas_errors), dtype=int)
        am = np.hstack((am1, am2))
        assert am.shape == (self.k, num_errors)

        # Construct p
        p1 = np.concatenate([dq_error_rate] * num_round)
        p2 = np.concatenate([meas_error_rate] * (num_round - 1))
        p = np.concatenate((p1, p2))
        assert p.shape == (num_errors,)

        return cm, am, p
