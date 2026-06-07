"""Helpers for building QEC circuits from YAML config blocks.

On import, sets the ``load_dir`` class-attribute for stim-file-backed circuits so
that ``create_circuit_from_config(...)`` works for them.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import qecdec
from qecdec.circuits import BB_144_12_12_Circuit, QECCircuit

# Path to the circuits directory at repo root.
CIRCUITS_ROOT = Path(__file__).resolve().parents[3] / "circuits"

# Point stim-file-backed circuit classes at their pre-generated .stim files.
BB_144_12_12_Circuit.load_dir = CIRCUITS_ROOT / "BB_144_12_12_Circuit"


def create_circuit_from_config(circuit_cfg: DictConfig) -> QECCircuit:
    """Build a QEC circuit from a YAML ``circuit`` config block.

    Expects ``circuit_cfg.circuit_name``, ``circuit_cfg.circuit_params``, and
    ``circuit_cfg.error_rate``.
    """
    circuit_params = OmegaConf.to_container(circuit_cfg.circuit_params, resolve=True)
    return qecdec.circuits.create_circuit_with_uniform_error_rate(
        circuit_cfg.circuit_name,
        circuit_cfg.error_rate,
        **circuit_params,
    )


def circuit_slug(circuit_cfg: DictConfig) -> str:
    """Directory slug for one ``(circuit_name, circuit_params)`` selection, e.g.
    ``basis=Z_d=5_rounds=5``. Params are sorted by key so the slug is stable.
    """
    circuit_params = OmegaConf.to_container(circuit_cfg.circuit_params, resolve=True)
    return "_".join(f"{k}={circuit_params[k]}" for k in sorted(circuit_params))
