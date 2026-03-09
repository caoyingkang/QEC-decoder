"""Utility functions and constants."""
from typing import Iterable
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

PYTORCH_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PYTORCH_ROOT / "runs"
BENCHMARK_ROOT = PYTORCH_ROOT / "benchmark"

BENCHMARK_CSV_FILENAME = "benchmark.csv"


def is_unique(items: Iterable) -> bool:
    """Check if an iterable contains unique elements."""
    seen = set()
    for x in items:
        if x in seen:
            return False
        seen.add(x)
    return True


def load_run_config(run_dir: Path) -> DictConfig:
    """Load config.yaml from `run_dir`."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return OmegaConf.load(config_path)


def is_consistent(run_dir: Path, code: str, noise_model: str, d: int, rounds: int, basis: str) -> bool:
    """
    Check if the QEC config of the run_dir agrees with the given code, noise model, d, rounds, and basis.
    """
    cfg = load_run_config(run_dir)
    qec = cfg.qec
    return (
        qec.code == code
        and qec.noise_model == noise_model
        and qec.d == d
        and qec.rounds == rounds
        and qec.basis == basis
    )


def extract_pytorch_decoder_name(run_dir: Path) -> str:
    """
    Extract the name of the PyTorch decoder from the run_dir.
    For example, 'pytorch/runs/RotatedSurfaceCode_Phenomenological/d=5_rounds=5_basis=Z/LearnedDMemBP/run_0' -> 'LearnedDMemBP/run_0'.
    """
    return "/".join(run_dir.parts[-2:])
