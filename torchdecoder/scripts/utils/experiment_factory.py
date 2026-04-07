from pathlib import Path

from omegaconf import DictConfig
from qecdec.experiments import RotatedSurfaceCode_Memory, StimFileExperiment
from qecdec.experiments.base import Experiment

# Path to the circuits directory at repo root
CIRCUITS_ROOT = Path(__file__).resolve().parents[3] / "circuits"


def get_stim_dir(qec_cfg: DictConfig) -> Path:
    """Infer the stim circuit directory from qec config fields."""
    return (
        CIRCUITS_ROOT
        / f"{qec_cfg.code}_{qec_cfg.noise_model}"
        / f"d={qec_cfg.d}_rounds={qec_cfg.rounds}_basis={qec_cfg.basis}"
    )


def create_experiment(qec_cfg: DictConfig, p: float) -> Experiment:
    """Create a QEC experiment from config and noise level p.

    For RotatedSurfaceCode, builds the experiment programmatically.
    For all other codes, loads from a stim circuit file inferred from the config fields:
        circuits/{code}_{noise_model}/d={d}_rounds={rounds}_basis={basis}/error_rate={p}.stim
    """
    if qec_cfg.code == "RotatedSurfaceCode":
        if qec_cfg.noise_model == "Phenomenological":
            return RotatedSurfaceCode_Memory(
                d=qec_cfg.d,
                rounds=qec_cfg.rounds,
                basis=qec_cfg.basis,
                data_qubit_error_rate=p,
                meas_error_rate=p,
            )
        else:
            raise ValueError(
                f"Unsupported noise model for RotatedSurfaceCode: {qec_cfg.noise_model}"
            )
    else:
        circuit_file = get_stim_dir(qec_cfg) / f"error_rate={p}.stim"
        return StimFileExperiment.load_from_file(circuit_file)
