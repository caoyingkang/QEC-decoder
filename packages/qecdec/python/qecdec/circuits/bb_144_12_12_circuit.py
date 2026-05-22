from pathlib import Path
from typing import ClassVar, Literal, Optional

import numpy as np
import stim

from .base import QECCircuit


def _detect_data_qubits(circuit: stim.Circuit) -> list[int]:
    """Detect data qubits as those that are only measured once in a circuit.
    This function is copied from https://github.com/trmue/relay/blob/main/src/relay_bp/stim/noise.py

    Warning: This is hacky and likely will only work with your typical memory circuits.
    """
    qubit_times_measured = [0 for qubit in range(circuit.num_qubits)]

    for inst in circuit:
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for qubit in inst.targets_copy():
                qubit_times_measured[qubit.qubit_value] += 1

    return [
        qubit
        for qubit, times_measured in enumerate(qubit_times_measured)
        if times_measured == 1
    ]


def _filter_detectors_by_basis(
    circuit: stim.Circuit,
    basis: str,
    qubits: list[int] | None = None,
) -> stim.Circuit | tuple[stim.Circuit, list[str]]:
    """Return a new circuit filtering any detectors which do not detect the specified basis for the input qubits.
    This function is copied from https://github.com/trmue/relay/blob/main/tests/testdata/utils.py

    Args:
        circuit: The original circuit
        basis: "X" or "Z"
        qubits: Data qubits to inject test errors on. Should typically be data qubits. Defaults
            to automatically detected data qubits which may not be robust.

    returns:
        The filtered circuit
    """
    assert basis in ("X", "Z")

    pauli_error = "Z" if basis == "X" else "X"

    circuit = circuit.flattened()

    noiseless_circuit = circuit.without_noise()
    sampler = noiseless_circuit.compile_detector_sampler()
    reference_detectors, reference_observables = sampler.sample(
        1, separate_observables=True
    )
    reference_detectors = reference_detectors[0, :]
    reference_observables = reference_observables[0, :]
    num_detectors = len(reference_detectors)

    detector_is_sensitive = np.full(num_detectors, False, dtype=bool)

    if qubits is None:
        to_test = _detect_data_qubits(noiseless_circuit)
    else:
        to_test = qubits

    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            injected_circuit = stim.Circuit()
            injected_circuit += noiseless_circuit
            injected_circuit.insert(
                inst_idx,
                stim.CircuitInstruction(f"{pauli_error}_ERROR", [qubit], [1.0]),
            )

            injected_sampler = injected_circuit.compile_detector_sampler()
            injected_detectors, injected_observables = injected_sampler.sample(
                1, separate_observables=True
            )
            injected_detectors = injected_detectors[0, :]
            injected_observables = injected_observables[0, :]

            detectors_flipped = np.where(reference_detectors != injected_detectors)
            detector_is_sensitive[detectors_flipped] = True

        to_test = []
        for inst in noiseless_circuit[inst_idx:]:
            # Is a reset we must inject errors after
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    filtered_circuit = stim.Circuit()
    detector_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            to_insert = detector_is_sensitive[detector_idx]
            detector_idx += 1
            if not to_insert:
                continue
        filtered_circuit.append(inst)
    return filtered_circuit


class BB_144_12_12_Circuit(QECCircuit, registry_name="BB_144_12_12_Circuit"):
    """Memory circuit for the gross code (i.e., a BB code with parameters [[144,12,12]])."""

    load_dir: ClassVar[Optional[Path]] = None

    def __init__(
        self,
        *,
        basis: Literal["X", "Z"],
        rounds: int,
        error_rate: float,
        filter_detectors: bool = True,
    ):
        """
        Load circuit at ``load_dir/basis=<basis>_rounds=<rounds>/error_rate=<error_rate>.stim``.
        If ``filter_detectors`` is True, filter out any off-basis detectors.

        User should set the class-attribute ``load_dir`` before instantiating this class.
        """
        if BB_144_12_12_Circuit.load_dir is None:
            raise RuntimeError(
                "Please set `BB_144_12_12_Circuit.load_dir` before instantiating this class"
            )

        path = (
            Path(BB_144_12_12_Circuit.load_dir)
            / f"basis={basis}_rounds={rounds}"
            / f"error_rate={error_rate}.stim"
        )
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        circuit = stim.Circuit.from_file(path)
        if filter_detectors:
            circuit = _filter_detectors_by_basis(circuit, basis)
        super().__init__(circuit)

    @classmethod
    def with_uniform_error_rate(
        cls,
        error_rate: float,
        *,
        basis: Literal["X", "Z"],
        rounds: int,
        filter_detectors: bool = True,
    ) -> "BB_144_12_12_Circuit":
        return BB_144_12_12_Circuit(
            basis=basis,
            rounds=rounds,
            error_rate=error_rate,
            filter_detectors=filter_detectors,
        )
