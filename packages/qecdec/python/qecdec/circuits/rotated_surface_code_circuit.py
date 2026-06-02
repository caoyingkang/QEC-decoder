from typing import Literal
from typing_extensions import Self

import stim

from .base import QECCircuit


class RotatedSurfaceCode_Circuit(
    QECCircuit, registry_name="RotatedSurfaceCode_Circuit"
):
    """Memory circuit for the rotated surface code."""

    def __init__(
        self,
        *,
        d: int,
        rounds: int,
        basis: Literal["X", "Z"],
        data_qubit_error_rate: float,
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
            basis : Literal['X', 'Z']
                Basis of logical state preparation and measurement. If basis='X'
                (resp. 'Z'), then we will use X-type (resp. Z-type) stabilizer
                measurement outcomes to correct Pauli Z (resp. X) errors.
            data_qubit_error_rate : float
                Error rate of data qubits before each round of stabilizer measurement.
            prep_error_rate : float
                Error rate of state preparation.
            meas_error_rate : float
                Error rate of measurement.
            gate1_error_rate : float
                Error rate of single-qubit gates.
            gate2_error_rate : float
                Error rate of two-qubit gates.
        """
        if d % 2 == 0:
            raise ValueError("Distance d must be an odd number")
        if d < 3:
            raise ValueError("Distance d must be at least 3")
        if rounds < 2:
            raise ValueError("rounds must be at least 2")
        if basis not in ["X", "Z"]:
            raise ValueError("Basis must be 'X' or 'Z'")

        self.d = d
        self.rounds = rounds
        self.basis = basis
        self.data_qubit_error_rate = data_qubit_error_rate
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate
        self.gate1_error_rate = gate1_error_rate
        self.gate2_error_rate = gate2_error_rate

        self.w = 2 * d + 1  # width of the grid holding the qubits
        self.num_dq = d * d  # number of data qubits
        self.num_xmq = (d * d - 1) // 2  # number of X-type measure qubits
        self.num_zmq = (d * d - 1) // 2  # number of Z-type measure qubits
        self.num_mq = self.num_xmq + self.num_zmq  # total number of measure qubits
        self.num_qubits = self.num_dq + self.num_mq  # total number of physical qubits

        # Lattice site coordinates of data qubits and measure qubits.
        self.dq_coos = frozenset(
            (x, y)
            for x in range(self.w)
            for y in range(self.w)
            if self._is_data_qubit_coord(x, y)
        )
        self.xmq_coos = frozenset(
            (x, y)
            for x in range(self.w)
            for y in range(self.w)
            if self._is_x_meas_qubit_coord(x, y)
        )
        self.zmq_coos = frozenset(
            (x, y)
            for x in range(self.w)
            for y in range(self.w)
            if self._is_z_meas_qubit_coord(x, y)
        )
        self.mq_coos = self.xmq_coos | self.zmq_coos
        assert len(self.dq_coos) == self.num_dq
        assert len(self.xmq_coos) == self.num_xmq
        assert len(self.zmq_coos) == self.num_zmq
        assert len(self.mq_coos) == self.num_mq

        # Lattice site indices of data qubits and measure qubits (sorted in ascending order).
        self.dq_inds = sorted(self._coo2ind(*coo) for coo in self.dq_coos)
        self.xmq_inds = sorted(self._coo2ind(*coo) for coo in self.xmq_coos)
        self.zmq_inds = sorted(self._coo2ind(*coo) for coo in self.zmq_coos)
        self.mq_inds = sorted(self.xmq_inds + self.zmq_inds)

        circuit = self._build_circuit()
        super().__init__(circuit)

    @classmethod
    def with_uniform_error_rate(
        cls,
        error_rate: float,
        *,
        d: int,
        rounds: int,
        basis: Literal["X", "Z"],
    ) -> Self:
        return RotatedSurfaceCode_Circuit(
            d=d,
            rounds=rounds,
            basis=basis,
            data_qubit_error_rate=error_rate,
            prep_error_rate=error_rate,
            meas_error_rate=error_rate,
            gate1_error_rate=error_rate,
            gate2_error_rate=error_rate,
        )

    def _build_circuit(self) -> stim.Circuit:
        # ------------------------------------------------------------------------------------------------
        # Build syndrome extraction circuit.
        # ------------------------------------------------------------------------------------------------
        circuit_SE = stim.Circuit()
        # Prepare all measure qubits in the |0> state.
        circuit_SE.append("R", self.mq_inds)
        circuit_SE.append("X_ERROR", self.mq_inds, self.prep_error_rate)
        circuit_SE.append("TICK")
        # Apply Hadamard gates to X-type measure qubits.
        circuit_SE.append("H", self.xmq_inds)
        circuit_SE.append("DEPOLARIZE1", self.xmq_inds, self.gate1_error_rate)
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 1st layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x < self.w - 1 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x, y), self._coo2ind(x + 1, y + 1)]
        for x, y in self.zmq_coos:
            if x < self.w - 1 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x + 1, y + 1), self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 2nd layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x > 0 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x, y), self._coo2ind(x - 1, y + 1)]
        for x, y in self.zmq_coos:
            if x < self.w - 1 and y > 0:
                cnot_indices += [self._coo2ind(x + 1, y - 1), self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 3rd layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x < self.w - 1 and y > 0:
                cnot_indices += [self._coo2ind(x, y), self._coo2ind(x + 1, y - 1)]
        for x, y in self.zmq_coos:
            if x > 0 and y < self.w - 1:
                cnot_indices += [self._coo2ind(x - 1, y + 1), self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)
        circuit_SE.append("TICK")
        # Apply CNOT gates in the 4th layer.
        cnot_indices = []
        for x, y in self.xmq_coos:
            if x > 0 and y > 0:
                cnot_indices += [self._coo2ind(x, y), self._coo2ind(x - 1, y - 1)]
        for x, y in self.zmq_coos:
            if x > 0 and y > 0:
                cnot_indices += [self._coo2ind(x - 1, y - 1), self._coo2ind(x, y)]
        circuit_SE.append("CNOT", cnot_indices)
        circuit_SE.append("DEPOLARIZE2", cnot_indices, self.gate2_error_rate)
        circuit_SE.append("TICK")
        # Apply Hadamard gates to X-type measure qubits.
        circuit_SE.append("H", self.xmq_inds)
        circuit_SE.append("DEPOLARIZE1", self.xmq_inds, self.gate1_error_rate)
        circuit_SE.append("TICK")
        # Readout all measure qubits.
        circuit_SE.append("X_ERROR", self.mq_inds, self.meas_error_rate)
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
        circuit_first_round.append("X_ERROR", self.dq_inds, self.prep_error_rate)
        circuit_first_round.append("TICK")
        # If basis='X', apply Hadamard gates to all data qubits.
        if self.basis == "X":
            circuit_first_round.append("H", self.dq_inds)
            circuit_first_round.append(
                "DEPOLARIZE1", self.dq_inds, self.gate1_error_rate
            )
            circuit_first_round.append("TICK")
        # Data qubits suffer from noise.
        circuit_first_round.append(
            "DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate
        )
        circuit_first_round.append("TICK")
        # Syndrome extraction.
        circuit_first_round += circuit_SE
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            x, y = self._ind2coo(i)
            if (self.basis == "Z" and self._is_z_meas_qubit_coord(x, y)) or (
                self.basis == "X" and self._is_x_meas_qubit_coord(x, y)
            ):
                circuit_first_round.append(
                    "DETECTOR", [stim.target_rec(-self.num_mq + k)], (x, y, 0)
                )

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for subsequent rounds.
        # ------------------------------------------------------------------------------------------------
        circuit_subsequent_round = stim.Circuit()
        # Data qubits suffer from noise.
        circuit_subsequent_round.append(
            "DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate
        )
        circuit_subsequent_round.append("TICK")
        # Syndrome extraction.
        circuit_subsequent_round += circuit_SE
        # Specify detectors.
        circuit_subsequent_round.append("SHIFT_COORDS", [], (0, 0, 1))
        for k, i in enumerate(self.mq_inds):
            x, y = self._ind2coo(i)
            if (self.basis == "Z" and self._is_z_meas_qubit_coord(x, y)) or (
                self.basis == "X" and self._is_x_meas_qubit_coord(x, y)
            ):
                circuit_subsequent_round.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-self.num_mq + k),
                        stim.target_rec(-self.num_mq * 2 + k),
                    ],
                    (x, y, 0),
                )

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for the final logical measurement.
        # ------------------------------------------------------------------------------------------------
        circuit_final_measurement = stim.Circuit()
        # If basis='X', apply Hadamard gates to all data qubits.
        if self.basis == "X":
            circuit_final_measurement.append("H", self.dq_inds)
            circuit_final_measurement.append(
                "DEPOLARIZE1", self.dq_inds, self.gate1_error_rate
            )
            circuit_final_measurement.append("TICK")
        # Measure all data qubits.
        circuit_final_measurement.append("X_ERROR", self.dq_inds, self.meas_error_rate)
        circuit_final_measurement.append("M", self.dq_inds)
        circuit_final_measurement.append("TICK")
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            x, y = self._ind2coo(i)
            if (self.basis == "Z" and self._is_z_meas_qubit_coord(x, y)) or (
                self.basis == "X" and self._is_x_meas_qubit_coord(x, y)
            ):
                lookback_indices = []
                if x < self.w - 1 and y < self.w - 1:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x + 1, y + 1))
                    )
                if x > 0 and y < self.w - 1:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x - 1, y + 1))
                    )
                if x < self.w - 1 and y > 0:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x + 1, y - 1))
                    )
                if x > 0 and y > 0:
                    lookback_indices.append(
                        -self.num_dq + self.dq_inds.index(self._coo2ind(x - 1, y - 1))
                    )
                lookback_indices.append(-self.num_dq - self.num_mq + k)
                circuit_final_measurement.append(
                    "DETECTOR",
                    [stim.target_rec(lb) for lb in lookback_indices],
                    (x, y, 1),
                )
        # Specify the logical measurement outcome.
        if self.basis == "Z":
            circuit_final_measurement.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(-self.num_dq + i) for i in range(self.d)],
                0,
            )
        else:
            circuit_final_measurement.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(-self.num_dq + i * self.d) for i in range(self.d)],
                0,
            )

        # ------------------------------------------------------------------------------------------------
        # Combine all the circuits.
        # ------------------------------------------------------------------------------------------------
        circuit = (
            circuit_first_round
            + circuit_subsequent_round * (self.rounds - 1)
            + circuit_final_measurement
        )

        return circuit

    def _is_data_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of a data qubit."""
        return 0 <= x < self.w and 0 <= y < self.w and x % 2 == 1 and y % 2 == 1

    def _is_x_meas_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of an X-type measure qubit."""
        return (
            2 <= x < self.w - 2
            and 0 <= y < self.w
            and x % 2 == 0
            and y % 2 == 0
            and (x + y) % 4 == 2
        )

    def _is_z_meas_qubit_coord(self, x: int, y: int) -> bool:
        """Check if (x, y) is the coordinate of a Z-type measure qubit."""
        return (
            0 <= x < self.w
            and 2 <= y < self.w - 2
            and x % 2 == 0
            and y % 2 == 0
            and (x + y) % 4 == 0
        )

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
