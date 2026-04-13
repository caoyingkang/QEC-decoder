from pathlib import Path

from qecdec.experiments import RotatedSurfaceCode_Memory, StimFileExperiment
from qecdec.experiments.base import Experiment

# Path to the circuits directory at repo root
CIRCUITS_ROOT = Path(__file__).resolve().parents[3] / "circuits"


def get_circuit_dir(
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
) -> Path:
    return (
        CIRCUITS_ROOT / f"{code}_{noise_model}" / f"d={d}_rounds={rounds}_basis={basis}"
    )


def create_experiment(
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
    p: float,
    load_circuit_from_file: bool,
) -> Experiment:
    """Create a QEC experiment from the given parameters."""
    if load_circuit_from_file:
        circuit_file = (
            get_circuit_dir(code, noise_model, d, rounds, basis)
            / f"error_rate={p}.stim"
        )
        return StimFileExperiment.load_from_file(circuit_file, detector_basis=basis)
    else:
        if code == "RotatedSurfaceCode" and noise_model == "Phenomenological":
            return RotatedSurfaceCode_Memory(
                d=d,
                rounds=rounds,
                basis=basis,
                data_qubit_error_rate=p,
                meas_error_rate=p,
            )
        else:
            raise NotImplementedError(
                f"Unsupported combination: {code} + {noise_model}"
            )
