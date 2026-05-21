from typing import Optional

import stim

from .base import QECCircuit


class RepetitionCode_Circuit(QECCircuit, registry_name="RepetitionCode_Circuit"):
    """Memory circuit for the repetition code."""

    def __init__(
        self,
        *,
        d: int,
        rounds: int,
        data_qubit_error_rate: Optional[float] = None,
        prep_error_rate: Optional[float] = None,
        meas_error_rate: Optional[float] = None,
        cnot_error_rate: Optional[float] = None,
    ):
        """
        Parameters
        ----------
            d : int
                Code distance.
            rounds : int
                Number of rounds of stabilizer measurement.
            data_qubit_error_rate : float or None
                Error rate of data qubits before each round of stabilizer measurement.
                If None, no data qubit error is included.
            prep_error_rate : float or None
                Error rate of state preparation. If None, no state preparation error is included.
            meas_error_rate : float or None
                Error rate of measurement. If None, no measurement error is included.
            cnot_error_rate : float or None
                Error rate of CNOT gates. If None, no CNOT gate error is included.
        """
        self.d = d
        self.rounds = rounds
        self.data_qubit_error_rate = data_qubit_error_rate
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate
        self.cnot_error_rate = cnot_error_rate

        self.num_dq = d  # number of data qubits
        self.num_mq = d - 1  # number of (Z-type) measure qubits
        self.num_qubits = self.num_dq + self.num_mq  # total number of physical qubits

        # Indices of data qubits and measure qubits.
        self.dq_inds = list(range(0, 2 * d, 2))  # 0, 2, 4, ..., 2d-2
        self.mq_inds = list(range(1, 2 * d - 1, 2))  # 1, 3, 5, ..., 2d-3

        circuit = self._build_circuit()
        super().__init__(circuit)

    @classmethod
    def with_uniform_error_rate(
        cls,
        error_rate: float,
        *,
        d: int,
        rounds: int,
    ) -> "RepetitionCode_Circuit":
        return RepetitionCode_Circuit(
            d=d,
            rounds=rounds,
            data_qubit_error_rate=error_rate,
            prep_error_rate=error_rate,
            meas_error_rate=error_rate,
            cnot_error_rate=error_rate,
        )

    def _build_circuit(self) -> stim.Circuit:
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
            circuit_first_round.append(
                "DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate
            )  # noqa: E501
        circuit_first_round.append("TICK")
        # Syndrome extraction.
        circuit_first_round += circuit_SE
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            circuit_first_round.append(
                "DETECTOR", [stim.target_rec(-self.num_mq + k)], (i, 0)
            )  # noqa: E501

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for subsequent rounds.
        # ------------------------------------------------------------------------------------------------
        circuit_subsequent_round = stim.Circuit()
        # Data qubits suffer from noise.
        if self.data_qubit_error_rate is not None:
            circuit_subsequent_round.append(
                "DEPOLARIZE1", self.dq_inds, self.data_qubit_error_rate
            )  # noqa: E501
        circuit_subsequent_round.append("TICK")
        # Syndrome extraction.
        circuit_subsequent_round += circuit_SE
        # Specify detectors.
        circuit_subsequent_round.append("SHIFT_COORDS", [], (0, 1))
        for k, i in enumerate(self.mq_inds):
            circuit_subsequent_round.append(
                "DETECTOR",
                [
                    stim.target_rec(-self.num_mq + k),
                    stim.target_rec(-self.num_mq * 2 + k),
                ],
                (i, 0),
            )

        # ------------------------------------------------------------------------------------------------
        # Build the circuit for the final logical measurement.
        # ------------------------------------------------------------------------------------------------
        circuit_final_measurement = stim.Circuit()
        # Readout all data qubits.
        if self.meas_error_rate is not None:
            circuit_final_measurement.append(
                "X_ERROR", self.dq_inds, self.meas_error_rate
            )  # noqa: E501
        circuit_final_measurement.append("M", self.dq_inds)
        circuit_final_measurement.append("TICK")
        # Specify detectors.
        for k, i in enumerate(self.mq_inds):
            circuit_final_measurement.append(
                "DETECTOR",
                [
                    stim.target_rec(-self.num_qubits + k),
                    stim.target_rec(-self.num_dq + k),
                    stim.target_rec(-self.num_dq + k + 1),
                ],
                (i, 1),
            )
        # Specify the logical measurement outcome.
        circuit_final_measurement.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(-self.num_dq)], 0
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
