from pathlib import Path
from collections import defaultdict
from typing import Any

from omegaconf import OmegaConf, DictConfig

from utils import is_unique
from bench.params import QECParams


def load_config_from_run_dir(run_dir: Path) -> DictConfig:
    """Load config.yaml from `run_dir`."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(
            f"Expected a DictConfig in {config_path}, but got {type(cfg).__name__}"
        )
    return cfg


def load_model_config_from_run_dir(run_dir: Path) -> DictConfig:
    """Load model config from `run_dir`."""
    cfg = load_config_from_run_dir(run_dir)
    if "model" not in cfg:
        raise ValueError("Model config does not exist.")
    model_cfg = cfg.model
    if not isinstance(model_cfg, DictConfig):
        raise ValueError(
            f"Expected a DictConfig for model_cfg, but got {type(model_cfg).__name__}"
        )
    return model_cfg


def _group_run_dirs_by_code_and_noise(
    run_dirs: list[Path],
) -> defaultdict[tuple[str, str], list[Path]]:
    """
    Split run_dirs into groups according to (code, noise_model) pairs.
    """
    grouped = defaultdict[tuple[str, str], list[Path]](list)
    for run_dir in run_dirs:
        cfg = load_config_from_run_dir(run_dir)
        grouped[(cfg.qec.code, cfg.qec.noise_model)].append(run_dir)
    return grouped


def _group_run_dirs_by_d_rounds_basis(
    run_dirs: list[Path],
) -> defaultdict[tuple[int, int, str], list[Path]]:
    """
    Split run_dirs into groups according to (d, rounds, basis) triples.
    """
    grouped = defaultdict[tuple[int, int, str], list[Path]](list)
    for run_dir in run_dirs:
        cfg = load_config_from_run_dir(run_dir)
        grouped[(cfg.qec.d, cfg.qec.rounds, cfg.qec.basis)].append(run_dir)
    return grouped


def discover_run_dirs(torch_runs_root: Path, qec_params: QECParams) -> list[Path]:
    """
    Discover all run directories (i.e., subdirectories of `torch_runs_root`
    that contain a `checkpoints/best_model.ckpt` file) that match the given
    QEC parameters.
    """
    all_run_dirs = [
        p.parent.parent for p in torch_runs_root.rglob("checkpoints/best_model.ckpt")
    ]
    if not is_unique(all_run_dirs):
        raise Exception("Duplicate run_dirs found.")

    run_dirs: list[Path] = []
    if len(all_run_dirs) > 0:
        grouped = _group_run_dirs_by_code_and_noise(all_run_dirs)
        key = (qec_params.code, qec_params.noise_model)
        if key in grouped:
            grouped = _group_run_dirs_by_d_rounds_basis(grouped[key])
            key = (qec_params.d, qec_params.rounds, qec_params.basis)
            if key in grouped:
                run_dirs = grouped[key]
    return run_dirs


def group_run_dirs_by_decoder_model_name(
    run_dirs: list[Path],
) -> defaultdict[str, list[Path]]:
    """
    Split run_dirs into groups according to decoder model name.
    """
    grouped = defaultdict[str, list[Path]](list)
    for run_dir in run_dirs:
        cfg = load_config_from_run_dir(run_dir)
        grouped[cfg.model.name].append(run_dir)
    return grouped


def extract_pytorch_decoder_run_id(run_dir: Path) -> int:
    """
    Extract the PyTorch decoder run id from the run_dir.
    For example, '.../LearnedDMemBP/run_0' -> 0.
    """
    return int(run_dir.name.split("_")[-1])


def extract_pytorch_decoder_name(run_dir: Path) -> str:
    """
    Extract the name of the PyTorch decoder from the run_dir.
    For example, '.../LearnedDMemBP/run_0' -> 'LearnedDMemBP/run_0'.
    """
    return "/".join(run_dir.parts[-2:])


def flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Recursively flatten nested config to dotted keys (e.g. model.num_iters, optim.lr_scheduler.factor).
    """
    result: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_config(v, key))
        else:
            result[key] = v
    return result


def get_differing_keys(configs: list[dict[str, Any]]) -> set[str]:
    """
    Given a list of flattened config dicts, return the set of keys whose values differ across configs.
    The list `configs` must contain at least two configs; raise an error if it does not.
    """
    n = len(configs)
    if n < 2:
        raise ValueError("List `configs` must contain at least two configs.")
    all_keys = set[str]()
    for cfg in configs:
        all_keys.update(cfg.keys())
    diff_keys = set[str]()
    for key in all_keys:
        values = [cfg[key] for cfg in configs if key in cfg]
        if len(values) < n:  # Some configs do not have this key.
            diff_keys.add(key)
            continue
        first = values[0]
        if not all(v == first for v in values):
            diff_keys.add(key)
    return diff_keys


def get_ckpt_path(run_dir: Path) -> Path:
    """
    Get the path to the checkpoint file from the run_dir.
    """
    return run_dir / "checkpoints" / "best_model.ckpt"
