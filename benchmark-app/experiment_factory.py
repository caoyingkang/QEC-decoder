from pathlib import Path

from qecdec.experiments import RotatedSurfaceCode_Memory, StimFileExperiment
from qecdec.experiments.base import Experiment

from bench.params import QECParams
from constants import CIRCUITS_ROOT


def _get_circuit_dir(
    qec_params: QECParams,
) -> Path:
    return (
        CIRCUITS_ROOT
        / f"{qec_params.code}_{qec_params.noise_model}"
        / f"d={qec_params.d}_rounds={qec_params.rounds}_basis={qec_params.basis}"
    )


def create_experiment(
    qec_params: QECParams,
    p: float,
    *,
    load_circuit_from_file: bool,
) -> Experiment:
    """Create a QEC experiment from the given parameters."""
    if load_circuit_from_file:
        circuit_file = _get_circuit_dir(qec_params) / f"error_rate={p}.stim"
        if qec_params.code.startswith("BB_"):
            return StimFileExperiment.load_from_file(
                circuit_file, detector_basis=qec_params.basis
            )
        elif qec_params.code == "HexColorCode":
            return StimFileExperiment.load_from_file(circuit_file)
        else:
            raise NotImplementedError(f"code={qec_params.code} not supported")
    else:
        code = qec_params.code
        noise_model = qec_params.noise_model
        if code == "RotatedSurfaceCode" and noise_model == "Phenomenological":
            return RotatedSurfaceCode_Memory(
                d=qec_params.d,
                rounds=qec_params.rounds,
                basis=qec_params.basis,
                data_qubit_error_rate=p,
                meas_error_rate=p,
            )
        elif code == "RotatedSurfaceCode" and noise_model == "CircuitLevel":
            return RotatedSurfaceCode_Memory(
                d=qec_params.d,
                rounds=qec_params.rounds,
                basis=qec_params.basis,
                data_qubit_error_rate=p,
                prep_error_rate=p,
                meas_error_rate=p,
                gate1_error_rate=p,
                gate2_error_rate=p,
            )
        else:
            raise NotImplementedError(
                f"Unsupported combination: {code} + {noise_model}"
            )
