from typing import Literal
from typing_extensions import Self

import stim

from .rotated_surface_code_base import RotatedSurfaceCodeBase


class RotatedSurfaceCode_Circuit(
    RotatedSurfaceCodeBase, registry_name="RotatedSurfaceCode_Circuit"
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
        self.data_qubit_error_rate = data_qubit_error_rate
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate
        self.gate1_error_rate = gate1_error_rate
        self.gate2_error_rate = gate2_error_rate
        super().__init__(d=d, rounds=rounds, basis=basis)

    @classmethod
    def with_uniform_error_rate(
        cls, error_rate: float, *, d: int, rounds: int, basis: Literal["X", "Z"]
    ) -> Self:
        return cls(
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
