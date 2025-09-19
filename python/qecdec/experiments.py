from .utils import extract_error_mechanisms_from_dem, extract_detector_coords_from_dem
from functools import cached_property, total_ordering
from dataclasses import dataclass
import numpy as np
import stim
from typing import Literal


@total_ordering
@dataclass(frozen=True)
class ErrorMechanism:
    """
    A dataclass representing an error mechanism.
    """
    dets: tuple[int, ...]  # flipped detectors (in increasing order), nonempty
    obsers: tuple[int, ...]  # flipped observables (in increasing order)
    p: float  # probability of occurrence
    start_layer: int  # the first layer in the decoding graph to find a flipped detector
    end_layer: int  # the last layer plus one in the decoding graph to find a flipped detector

    def __eq__(self, other) -> bool:
        """
        Two error mechanisms are considered equal if they flip the same set of detectors. In most cases, this 
        guarantees that they also flip the same set of observables, because otherwise the fault distance of the 
        circuit would be at most 2.
        """
        if not isinstance(other, ErrorMechanism):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        # The following guarantees that self.start_layer == other.start_layer and self.end_layer == other.end_layer
        return self.dets == other.dets

    def __lt__(self, other) -> bool:
        """
        To sort error mechanisms, we first compare `start_layer`, then `end_layer`, and finally `dets`.
        """
        if not isinstance(other, ErrorMechanism):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        if self.start_layer != other.start_layer:
            return self.start_layer < other.start_layer
        elif self.end_layer != other.end_layer:
            return self.end_layer < other.end_layer
        else:
            return self.dets < other.dets


class MemoryExperiment:
    """Base class for all memory experiments.
    """

    def __init__(
        self,
        rounds: int,
        num_detectors_per_layer: int,
        num_observables: int,
    ):
        self.rounds = rounds  # number of rounds of syndrome extraction
        self.layers = rounds + 1  # number of layers of detectors in the decoding graph
        self.num_detectors_per_layer = num_detectors_per_layer
        self.num_detectors = self.layers * num_detectors_per_layer
        self.num_observables = num_observables

    @cached_property
    def circuit(self) -> stim.Circuit:
        """
        stim.Circuit object representing the experiment.
        """
        raise NotImplementedError

    @cached_property
    def eid2emech(self) -> dict[int, ErrorMechanism]:
        """
        A dictionary mapping error ids to ErrorMechanism objects.
        """
        dem = self.circuit.detector_error_model()
        eff2prob = extract_error_mechanisms_from_dem(dem)

        emechs: list[ErrorMechanism] = []
        for (dets, obsers), p in eff2prob.items():
            assert len(dets) > 0
            start_layer = dets[0] // self.num_detectors_per_layer
            end_layer = dets[-1] // self.num_detectors_per_layer + 1
            emechs.append(ErrorMechanism(
                dets, obsers, p, start_layer, end_layer))
        emechs.sort()

        return {i: e for i, e in enumerate(emechs)}

    @property
    def num_error_mechanisms(self) -> int:
        """
        The number of error mechanisms.
        """
        return len(self.eid2emech)

    @cached_property
    def chkmat(self) -> np.ndarray:
        """
        Check matrix, shape=(#detectors, #error_mechanisms), dtype=np.uint8 ∈ {0,1}.
        (i, j) entry is 1 iff detector i is flipped by error mechanism j.
        """
        chkmat = np.zeros(
            (self.num_detectors, self.num_error_mechanisms), dtype=np.uint8)
        for j, e in self.eid2emech.items():
            chkmat[e.dets, j] = 1
        return chkmat

    @cached_property
    def obsmat(self) -> np.ndarray:
        """
        Observable matrix, shape=(#logical_qubits, #error_mechanisms), dtype=np.uint8 ∈ {0,1}.
        (i, j) entry is 1 iff observable i is flipped by error mechanism j.
        """
        obsmat = np.zeros(
            (self.num_observables, self.num_error_mechanisms), dtype=np.uint8)
        for j, e in self.eid2emech.items():
            obsmat[e.obsers, j] = 1
        return obsmat

    @cached_property
    def prior(self) -> np.ndarray:
        """
        Vector of prior probabilities for each error mechanism, shape=(#error_mechanisms,).
        """
        prior = np.zeros(self.num_error_mechanisms)
        for j, e in self.eid2emech.items():
            prior[j] = e.p
        return prior

    @cached_property
    def detector_coords(self) -> np.ndarray:
        """
        Array of detector coordinates, shape=(#detectors, #coordinates), dtype=float. This can be used to visualize the decoding graph.
        """
        dem = self.circuit.detector_error_model()
        return extract_detector_coords_from_dem(dem)

    @cached_property
    def error_coords(self) -> np.ndarray:
        """
        Array of error coordinates, shape=(#error_mechanisms, #coordinates), dtype=float. This can be used to visualize the decoding graph.
        """
        raise NotImplementedError


class RepetitionCode_Memory(MemoryExperiment):
    """Memory experiment for the repetition code.
    """

    def __init__(
        self,
        d: int,
        rounds: int,
        *,
        data_qubit_error_rate: float | None = None,
        prep_error_rate: float | None = None,
        meas_error_rate: float | None = None,
        cnot_error_rate: float | None = None,
    ):
        """
        Parameters
        ----------
            d : int
                The code distance.

            rounds : int
                The number of rounds of syndrome extraction.

            data_qubit_error_rate : float, optional
                The error rate of data qubits before each round of syndrome extraction. If None, no data qubit error is included.

            prep_error_rate : float, optional
                The error rate of state preparation. If None, no state preparation error is included.

            meas_error_rate : float, optional
                The error rate of measurement. If None, no measurement error is included.

            cnot_error_rate : float, optional
                The error rate of CNOT gates. If None, no CNOT gate error is included.
        """
        self.d = d
        self.num_dq = d  # number of data qubits
        self.num_mq = d - 1  # number of (Z-type) measure qubits
        self.num_qubits = self.num_dq + self.num_mq  # total number of physical qubits
        self.k = 1  # number of logical qubits

        super().__init__(
            rounds=rounds,
            num_detectors_per_layer=self.num_mq,
            num_observables=self.k
        )

        self.data_qubit_error_rate = data_qubit_error_rate
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate
        self.cnot_error_rate = cnot_error_rate

        # Indices of data qubits and measure qubits.
        self.dq_inds = list(range(0, 2 * d, 2))  # 0, 2, 4, ..., 2d-2
        self.mq_inds = list(range(1, 2 * d - 1, 2))  # 1, 3, 5, ..., 2d-3

    @cached_property
    def H(self) -> np.ndarray:
        """
        (Z-type) stabilizer matrix, shape=(#measure_qubits, #data_qubits), dtype=np.uint8 ∈ {0,1}.
        """
        H = np.zeros((self.num_mq, self.num_dq), dtype=np.uint8)
        for i in range(self.num_mq):
            H[i, i] = 1
            H[i, i + 1] = 1
        return H

    @cached_property
    def L(self) -> np.ndarray:
        """
        (Z-type) logical operator matrix, shape=(#logical_qubits, #data_qubits), dtype=np.uint8 ∈ {0,1}.
        """
        L = np.zeros((self.k, self.num_dq), dtype=np.uint8)
        L[0, 0] = 1
        return L

    @cached_property
    def circuit(self) -> stim.Circuit:
        # ------------------------------------------------------------------------------------------------
        # Build syndrome extraction circuit.
        # ------------------------------------------------------------------------------------------------
        circuit_SE = stim.Circuit()
        # Prepare all measure qubits in the |0> state.
        circuit_SE.append("R", self.mq_inds)
        if self.prep_error_rate is not None:
            circuit_SE.append("X_ERROR", self.mq_inds, self.prep_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 1st layer.
        cnot_inds = list(range(0, self.num_qubits - 1))
        circuit_SE.append("CNOT", cnot_inds)
        if self.cnot_error_rate is not None:
            circuit_SE.append("DEPOLARIZE2", cnot_inds, self.cnot_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 2nd layer.
        cnot_inds = list(range(self.num_qubits - 1, 0, -1))
        circuit_SE.append("CNOT", cnot_inds)
        if self.cnot_error_rate is not None:
            circuit_SE.append("DEPOLARIZE2", cnot_inds, self.cnot_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Readout all measure qubits.
        if self.meas_error_rate is not None:
            circuit_SE.append("X_ERROR", self.mq_inds, self.meas_error_rate)  # noqa: E501
        circuit_SE.append("M", self.mq_inds)
        circuit_SE.append("TICK")

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for the first round.
        # ------------------------------------------------------------------------------------------------
        circuit_first_round = stim.Circuit()
        # Prepare all data qubits in the |0> state.
        circuit_first_round.append("R", self.dq_inds)
        if self.prep_error_rate is not None:
            circuit_first_round.append("X_ERROR", self.dq_inds, self.prep_error_rate)  # noqa: E501
        circuit_first_round.append("TICK")
        # Data qubits suffer from noise.
        if self.data_qubit_error_rate is not None:
            circuit_first_round.append("DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate)  # noqa: E501
        circuit_first_round.append("TICK")
        # Syndrome extraction.
        circuit_first_round += circuit_SE
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            circuit_first_round.append("DETECTOR", [stim.target_rec(-self.num_mq + k)], (i, 0))  # noqa: E501

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for subsequent rounds.
        # ------------------------------------------------------------------------------------------------
        circuit_subsequent_round = stim.Circuit()
        # Data qubits suffer from noise.
        if self.data_qubit_error_rate is not None:
            circuit_subsequent_round.append("DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate)  # noqa: E501
        circuit_subsequent_round.append("TICK")
        # Syndrome extraction.
        circuit_subsequent_round += circuit_SE
        # Specify detectors.
        circuit_subsequent_round.append("SHIFT_COORDS", [], (0, 1))
        for k, i in enumerate(self.mq_inds):
            circuit_subsequent_round.append(
                "DETECTOR",
                [stim.target_rec(-self.num_mq + k),
                 stim.target_rec(-self.num_mq * 2 + k)],
                (i, 0)
            )

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for the final logical measurement.
        # ------------------------------------------------------------------------------------------------
        circuit_final_measurement = stim.Circuit()
        # Readout all data qubits.
        if self.meas_error_rate is not None:
            circuit_final_measurement.append("X_ERROR", self.dq_inds, self.meas_error_rate)  # noqa: E501
        circuit_final_measurement.append("M", self.dq_inds)
        circuit_final_measurement.append("TICK")
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            circuit_final_measurement.append(
                "DETECTOR",
                [stim.target_rec(-self.num_qubits + k),
                 stim.target_rec(-self.num_dq + k),
                 stim.target_rec(-self.num_dq + k + 1)],
                (i, 1)
            )
        # Specify the logical measurement outcome.
        circuit_final_measurement.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(-self.num_dq)],
            0
        )

        # ------------------------------------------------------------------------------------------------
        # Combine all the circuits.
        # ------------------------------------------------------------------------------------------------
        circuit = circuit_first_round + circuit_subsequent_round * \
            (self.rounds - 1) + circuit_final_measurement

        assert circuit.num_detectors == self.num_detectors
        assert circuit.num_observables == self.num_observables
        return circuit


class RotatedSurfaceCode_Memory(MemoryExperiment):
    """Memory experiment for the rotated surface code.
    """

    def __init__(
        self,
        d: int,
        rounds: int,
        *,
        basis: Literal['X', 'Z'],
        data_qubit_error_rate: float | None = None,
        prep_error_rate: float | None = None,
        meas_error_rate: float | None = None,
        gate1_error_rate: float | None = None,
        gate2_error_rate: float | None = None,
    ):
        """
        Parameters
        ----------
            d : int
                The code distance.

            rounds : int
                The number of rounds of syndrome extraction.

            basis : Literal['X', 'Z']
                The basis of logical state preparation and measurement. If basis='X' (resp. 'Z'), then we will use X-type (resp. Z-type) 
                stabilizer measurement outcomes to correct Pauli Z (resp. X) errors.

            data_qubit_error_rate : float, optional
                The error rate of data qubits before each round of syndrome extraction. If None, no data qubit error is included.

            prep_error_rate : float, optional
                The error rate of state preparation. If None, no state preparation error is included.

            meas_error_rate : float, optional
                The error rate of measurement. If None, no measurement error is included.

            gate1_error_rate : float, optional
                The error rate of single-qubit gates. If None, no single-qubit gate error is included.

            gate2_error_rate : float, optional
                The error rate of two-qubit gates. If None, no two-qubit gate error is included.
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be an odd number")
        if d < 3:
            raise ValueError("Distance d must be at least 3")
        if basis not in ['X', 'Z']:
            raise ValueError("Basis must be 'X' or 'Z'")

        self.d = d
        self.w = 2 * d + 1  # width of the grid holding the qubits
        self.num_dq = d * d  # number of data qubits
        self.num_xmq = (d * d - 1) // 2  # number of X-type measure qubits
        self.num_zmq = (d * d - 1) // 2  # number of Z-type measure qubits
        self.num_mq = self.num_xmq + self.num_zmq  # total number of measure qubits
        self.num_qubits = self.num_dq + self.num_mq  # total number of physical qubits
        self.k = 1  # number of logical qubits
        self.basis = basis

        super().__init__(
            rounds=rounds,
            num_detectors_per_layer=self.num_xmq if basis == 'X' else self.num_zmq,
            num_observables=self.k
        )

        self.data_qubit_error_rate = data_qubit_error_rate
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate
        self.gate1_error_rate = gate1_error_rate
        self.gate2_error_rate = gate2_error_rate

        # Lattice site coordinates of data qubits and measure qubits.
        self.dq_coos = frozenset((x, y) for x in range(self.w)
                                 for y in range(self.w)
                                 if self._is_data_qubit_coord(x, y))
        self.xmq_coos = frozenset((x, y) for x in range(self.w)
                                  for y in range(self.w)
                                  if self._is_x_meas_qubit_coord(x, y))
        self.zmq_coos = frozenset((x, y) for x in range(self.w)
                                  for y in range(self.w)
                                  if self._is_z_meas_qubit_coord(x, y))
        self.mq_coos = self.xmq_coos | self.zmq_coos
        assert len(self.dq_coos) == self.num_dq
        assert len(self.xmq_coos) == self.num_xmq
        assert len(self.zmq_coos) == self.num_zmq
        assert len(self.mq_coos) == self.num_mq

        # Lattice site indices of data qubits and measure qubits (sorted in ascending order).
        self.dq_inds = sorted(self._coo2ind(*coo)
                              for coo in self.dq_coos)
        self.xmq_inds = sorted(self._coo2ind(*coo)
                               for coo in self.xmq_coos)
        self.zmq_inds = sorted(self._coo2ind(*coo)
                               for coo in self.zmq_coos)
        self.mq_inds = sorted(self.xmq_inds + self.zmq_inds)

    @cached_property
    def circuit(self) -> stim.Circuit:
        # ------------------------------------------------------------------------------------------------
        # Build syndrome extraction circuit.
        # ------------------------------------------------------------------------------------------------
        circuit_SE = stim.Circuit()
        # Prepare all measure qubits in the |0> state.
        circuit_SE.append("R", self.mq_inds)
        if self.prep_error_rate is not None:
            circuit_SE.append("X_ERROR", self.mq_inds, self.prep_error_rate)
        circuit_SE.append("TICK")
        # Apply Hadamard gates to X-type measure qubits.
        circuit_SE.append("H", self.xmq_inds)
        if self.gate1_error_rate is not None:
            circuit_SE.append("DEPOLARIZE1", self.xmq_inds, self.gate1_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 1st layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x < self.w - 1 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x, y),
                                 self._coo2ind(x + 1, y + 1)]
        for x, y in self.zmq_coos:
            if x < self.w - 1 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x + 1, y + 1),
                                 self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        if self.gate2_error_rate is not None:
            circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 2nd layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x > 0 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x, y),
                                 self._coo2ind(x - 1, y + 1)]
        for x, y in self.zmq_coos:
            if x < self.w - 1 and y > 0:
                cnot_indices += [self._coo2ind(x + 1, y - 1),
                                 self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        if self.gate2_error_rate is not None:
            circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 3rd layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x < self.w - 1 and y > 0:
                cnot_indices += [self._coo2ind(x, y),
                                 self._coo2ind(x + 1, y - 1)]
        for x, y in self.zmq_coos:
            if x > 0 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x - 1, y + 1),
                                 self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        if self.gate2_error_rate is not None:
            circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 4th layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x > 0 and y > 0:
                cnot_indices += [self._coo2ind(x, y),
                                 self._coo2ind(x - 1, y - 1)]
        for x, y in self.zmq_coos:
            if x > 0 and y > 0:
                cnot_indices += [self._coo2ind(x - 1, y - 1),
                                 self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        if self.gate2_error_rate is not None:
            circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Apply Hadamard gates to X-type measure qubits.
        circuit_SE.append("H", self.xmq_inds)
        if self.gate1_error_rate is not None:
            circuit_SE.append("DEPOLARIZE1", self.xmq_inds, self.gate1_error_rate)  # noqa: E501
        circuit_SE.append("TICK")
        # Readout all measure qubits.
        if self.meas_error_rate is not None:
            circuit_SE.append("X_ERROR", self.mq_inds, self.meas_error_rate)  # noqa: E501
        circuit_SE.append("M", self.mq_inds)
        circuit_SE.append("TICK")

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for the first round.
        # ------------------------------------------------------------------------------------------------
        circuit_first_round = stim.Circuit()
        # Specify the coordinates of all qubits.
        for i in self.dq_inds + self.mq_inds:
            circuit_first_round.append("QUBIT_COORDS", i, self._ind2coo(i))
        # Prepare all data qubits in the |0> state.
        circuit_first_round.append("R", self.dq_inds)
        if self.prep_error_rate is not None:
            circuit_first_round.append("X_ERROR", self.dq_inds, self.prep_error_rate)  # noqa: E501
        circuit_first_round.append("TICK")
        # If basis='X', apply Hadamard gates to all data qubits.
        if self.basis == 'X':
            circuit_first_round.append("H", self.dq_inds)
            if self.gate1_error_rate is not None:
                circuit_first_round.append("DEPOLARIZE1", self.dq_inds, self.gate1_error_rate)  # noqa: E501
            circuit_first_round.append("TICK")
        # Data qubits suffer from noise.
        if self.data_qubit_error_rate is not None:
            circuit_first_round.append("DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate)  # noqa: E501
        circuit_first_round.append("TICK")
        # Syndrome extraction.
        circuit_first_round += circuit_SE
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            x, y = self._ind2coo(i)
            if (self.basis == 'Z' and self._is_z_meas_qubit_coord(x, y)) or \
                    (self.basis == 'X' and self._is_x_meas_qubit_coord(x, y)):
                circuit_first_round.append("DETECTOR", [stim.target_rec(-self.num_mq + k)], (x, y, 0))  # noqa: E501

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for subsequent rounds.
        # ------------------------------------------------------------------------------------------------
        circuit_subsequent_round = stim.Circuit()
        # Data qubits suffer from noise.
        if self.data_qubit_error_rate is not None:
            circuit_subsequent_round.append("DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate)  # noqa: E501
        circuit_subsequent_round.append("TICK")
        # Syndrome extraction.
        circuit_subsequent_round += circuit_SE
        # Specify detectors.
        circuit_subsequent_round.append("SHIFT_COORDS", [], (0, 0, 1))
        for k, i in enumerate(self.mq_inds):
            x, y = self._ind2coo(i)
            if (self.basis == 'Z' and self._is_z_meas_qubit_coord(x, y)) or \
                    (self.basis == 'X' and self._is_x_meas_qubit_coord(x, y)):
                circuit_subsequent_round.append(
                    "DETECTOR",
                    [stim.target_rec(-self.num_mq + k),
                     stim.target_rec(-self.num_mq * 2 + k)],
                    (x, y, 0)
                )

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for the final logical measurement.
        # ------------------------------------------------------------------------------------------------
        circuit_final_measurement = stim.Circuit()
        # If basis='X', apply Hadamard gates to all data qubits.
        if self.basis == 'X':
            circuit_final_measurement.append("H", self.dq_inds)
            if self.gate1_error_rate is not None:
                circuit_final_measurement.append("DEPOLARIZE1", self.dq_inds, self.gate1_error_rate)  # noqa: E501
            circuit_final_measurement.append("TICK")
        # Measure all data qubits.
        if self.meas_error_rate is not None:
            circuit_final_measurement.append("X_ERROR", self.dq_inds, self.meas_error_rate)  # noqa: E501
        circuit_final_measurement.append("M", self.dq_inds)
        circuit_final_measurement.append("TICK")
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            x, y = self._ind2coo(i)
            if (self.basis == 'Z' and self._is_z_meas_qubit_coord(x, y)) or \
                    (self.basis == 'X' and self._is_x_meas_qubit_coord(x, y)):
                lookback_indices = []
                if x < self.w - 1 and y < self.w - 1:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x + 1, y + 1)))
                if x > 0 and y < self.w - 1:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x - 1, y + 1)))
                if x < self.w - 1 and y > 0:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x + 1, y - 1)))
                if x > 0 and y > 0:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x - 1, y - 1)))
                lookback_indices.append(-self.num_dq - self.num_mq + k)
                circuit_final_measurement.append(
                    "DETECTOR",
                    [stim.target_rec(l) for l in lookback_indices],
                    (x, y, 1)
                )
        # Specify the logical measurement outcome.
        if self.basis == 'Z':
            circuit_final_measurement.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(-self.num_dq + i) for i in range(self.d)],
                0
            )
        else:
            circuit_final_measurement.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(-self.num_dq + i * self.d) for i in range(self.d)],  # noqa: E501
                0
            )

        # ------------------------------------------------------------------------------------------------
        # Combine all the circuits.
        # ------------------------------------------------------------------------------------------------
        circuit = circuit_first_round + circuit_subsequent_round * \
            (self.rounds - 1) + circuit_final_measurement

        assert circuit.num_detectors == self.num_detectors
        assert circuit.num_observables == self.num_observables
        return circuit

    @cached_property
    def error_coords(self) -> np.ndarray:
        error_coords = np.zeros((self.num_error_mechanisms,
                                 self.detector_coords.shape[1]))
        for i, e in self.eid2emech.items():
            dets = e.dets
            assert len(dets) > 0
            if len(dets) == 1:
                coo = self.detector_coords[dets[0]].tolist()
                assert len(coo) == 3
                x, y, t = coo
                if self.basis == 'Z':
                    if y < 2.001:
                        error_coords[i] = np.array([x, y - 1, t])
                    else:
                        error_coords[i] = np.array([x, y + 1, t])
                else:
                    if x < 2.001:
                        error_coords[i] = np.array([x - 1, y, t])
                    else:
                        error_coords[i] = np.array([x + 1, y, t])
            else:
                error_coords[i] = np.mean(
                    self.detector_coords[dets, :], axis=0)
        return error_coords

    def _is_data_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of a data qubit."""
        return 0 <= x < self.w and 0 <= y < self.w and x % 2 == 1 and y % 2 == 1

    def _is_x_meas_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of an X-type measure qubit."""
        return 2 <= x < self.w - 2 and 0 <= y < self.w and x % 2 == 0 and y % 2 == 0 and (x + y) % 4 == 2

    def _is_z_meas_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of a Z-type measure qubit."""
        return 0 <= x < self.w and 2 <= y < self.w - 2 and x % 2 == 0 and y % 2 == 0 and (x + y) % 4 == 0

    def _coo2ind(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to lattice site index."""
        if (x + y) % 2 == 1:
            raise ValueError("Not a valid lattice site")
        return (y // 2) * self.w + x

    def _ind2coo(self, i: int) -> tuple[int, int]:
        """Convert lattice site index to (x, y) coordinates."""
        x = i % self.w
        y = 2 * (i // self.w) + (x % 2)
        return x, y


__all__ = [
    "RepetitionCode_Memory",
    "RotatedSurfaceCode_Memory",
]
