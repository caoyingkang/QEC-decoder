from collections.abc import Mapping
from pathlib import Path
from typing import Any

from constants import RESULTS_DIR, TORCH_RUNS_ROOT


def dict_to_str(d: Mapping[str, Any], *, seperator: str = "_") -> str:
    return seperator.join(f"{k}={v}" for k, v in sorted(d.items()))


def get_csv_path(
    circuit_name: str, circuit_params: Mapping[str, Any], decoder_name: str
) -> Path:
    return (
        RESULTS_DIR
        / circuit_name
        / dict_to_str(circuit_params)
        / decoder_name
        / "results.csv"
    )


def discover_torch_run_dirs(
    circuit_name: str, circuit_params: dict[str, Any], model_name: str
) -> list[Path]:
    """
    Discover all run directories (i.e., subdirectories of `TORCH_RUNS_ROOT`
    that contain a `checkpoints/best_model.ckpt` file) that match the given
    ``circuit_name``, ``circuit_params``, and ``model_name``.
    """
    subdir = TORCH_RUNS_ROOT / circuit_name / dict_to_str(circuit_params) / model_name
    run_dirs = [p.parent.parent for p in subdir.rglob("checkpoints/best_model.ckpt")]
    if len(set(run_dirs)) != len(run_dirs):
        raise Exception("Duplicate run_dirs found.")
    return run_dirs
