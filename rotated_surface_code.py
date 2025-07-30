import numpy as np
import scipy.sparse as sp
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

        # Check orthogonality conditions
        assert np.all((self.Hx @ self.Hz.T).data % 2 == 0)
        assert np.all((self.Hx @ self.Lz.T).data % 2 == 0)
        assert np.all((self.Hz @ self.Lx.T).data % 2 == 0)
        assert np.all((self.Lx @ self.Lz.T).toarray() % 2 ==
                      np.eye(self.k, dtype=np.uint8))

    def get_check_matrix_action_matrix_probability_vector(self,
                                                          noise_model: Literal['code-capacity', 'phenomenological'],
                                                          detector_type: Literal['X', 'Z'],
                                                          **kwargs
                                                          ) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
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
            cm : csr_matrix
                Check matrix, shape=(#detectors, #error_mechanisms), dtype=np.uint8, values in {0, 1}.
                A nonzero entry at row i and column j indicates that the i-th detector is flipped by the j-th error mechanism.

            am : csr_matrix
                Action matrix, shape=(#logical_qubits, #error_mechanisms), dtype=np.uint8, values in {0, 1}.
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
        Construct the X- and Z-type stabilizer matrices self.Hx and self.Hz, CSR format, dtype=np.uint8, values in {0, 1}.
        The i-th X-type (resp. Z-type) stabilizer is the tensor product of Pauli X (resp. Z) operators acting on the data 
        qubits indexed by the the i-th row of self.Hx (resp. self.Hz).
        """
        # These are used to construct the CSR sparse matrices self.Hx and self.Hz
        Hx_indices, Hx_indptr = [], [0]
        Hz_indices, Hz_indptr = [], [0]

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
            Hx_indices.extend(data_qubits)
            Hx_indptr.append(len(Hx_indices))

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
            Hz_indices.extend(data_qubits)
            Hz_indptr.append(len(Hz_indices))

        # Create CSR sparse matrices
        Hx_data = np.ones(len(Hx_indices), dtype=np.uint8)
        Hz_data = np.ones(len(Hz_indices), dtype=np.uint8)
        self.Hx = sp.csr_matrix((Hx_data, Hx_indices, Hx_indptr),
                                shape=(self.mx, self.n))
        self.Hz = sp.csr_matrix((Hz_data, Hz_indices, Hz_indptr),
                                shape=(self.mz, self.n))

    def _construct_logical_operator_matrices(self):
        """Construct the matrices self.Lx and self.Lz representing logical X and Z operators, CSR format, dtype=np.uint8, values in {0, 1}.
        The i-th logical X (resp. Z) operator is the tensor product of Pauli X (resp. Z) operators acting on the data qubits indexed by the 
        i-th row of self.Hx (resp. self.Hz).
        """
        # These are used to construct the CSR sparse matrices self.Lx and self.Lz
        Lx_indices, Lx_indptr = [], [0]
        Lz_indices, Lz_indptr = [], [0]

        # logical X
        data_qubits = [self._coord_to_dq(row, 1)
                       for row in range(1, 2 * self.d, 2)]
        Lx_indices.extend(data_qubits)
        Lx_indptr.append(len(Lx_indices))

        # logical Z
        data_qubits = [self._coord_to_dq(1, col)
                       for col in range(1, 2 * self.d, 2)]
        Lz_indices.extend(data_qubits)
        Lz_indptr.append(len(Lz_indices))

        # Create CSR sparse matrices
        Lx_data = np.ones(len(Lx_indices), dtype=np.uint8)
        Lz_data = np.ones(len(Lz_indices), dtype=np.uint8)
        self.Lx = sp.csr_matrix((Lx_data, Lx_indices, Lx_indptr),
                                shape=(1, self.n))
        self.Lz = sp.csr_matrix((Lz_data, Lz_indices, Lz_indptr),
                                shape=(1, self.n))

    def _cm_am_p_phenomenological(
        self,
        detector_type: Literal['X', 'Z'],
        num_round: int,
        dq_error_rate: np.ndarray,
        meas_error_rate: np.ndarray
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
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
        cm1 = sp.block_diag([H] * num_round, format='csr')
        cm2_data = np.ones((2, num_meas_errors), dtype=np.uint8)
        cm2_offsets = [0, -m]
        cm2 = sp.dia_matrix((cm2_data, cm2_offsets),
                            shape=(num_detectors, num_meas_errors))
        cm = sp.hstack([cm1, cm2], format='csr')
        assert isinstance(cm, sp.csr_matrix)
        assert cm.shape == (num_detectors, num_errors)
        assert cm.dtype == np.uint8

        # Construct am
        am1 = sp.hstack([L] * num_round, format='csr')
        am2 = sp.csr_matrix((1, num_meas_errors),
                            dtype=np.uint8)
        am = sp.hstack([am1, am2], format='csr')
        assert isinstance(am, sp.csr_matrix)
        assert am.shape == (self.k, num_errors)
        assert am.dtype == np.uint8

        # Construct p
        p1 = np.concatenate([dq_error_rate] * num_round)
        p2 = np.concatenate([meas_error_rate] * (num_round - 1))
        p = np.concatenate((p1, p2))
        assert p.shape == (num_errors,)

        return cm, am, p
