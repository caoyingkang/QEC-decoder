"""Utility functions and constants."""
import html
import re
from typing import Any, Iterable
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


def highlight_yaml_differences(yaml_str: str, diff_keys: set[str]) -> str:
    """
    Wrap YAML lines that correspond to diff_keys in a highlight span.
    
    Uses <br> for line breaks and &nbsp; for leading spaces to preserve YAML indentation.
    """
    lines = yaml_str.split("\n")
    result: list[str] = []
    stack: list[str] = []  # path of parent keys at current indent

    def preserve_indentation(s: str, spaces_per_level: int = 4) -> str:
        """Replace leading spaces with &nbsp; and expand for better visibility."""
        stripped = s.lstrip(" ")
        n_spaces = len(s) - len(stripped)
        # Expand: each 2-space YAML level becomes spaces_per_level for display
        n_levels = n_spaces // 2
        expanded = n_levels * spaces_per_level
        return "&nbsp;" * expanded + stripped

    for line in lines:
        m = re.match(r"^(\s*)(\w+)\s*:", line)
        content = preserve_indentation(html.escape(line))
        if m:
            indent = len(m.group(1))
            depth = indent // 2
            key_name = m.group(2)
            # Pop stack to match current depth (we may have left a nested block)
            while len(stack) > depth:
                stack.pop()
            full_key = ".".join(stack) + "." + key_name if stack else key_name
            # Push this key for any nested content that follows
            stack.append(key_name)
            if full_key in diff_keys:
                result.append(f'<span style="background-color: #fff3cd">{content}</span>')
            else:
                result.append(content)
        else:
            result.append(content)
    pre_style = (
        "white-space: pre-wrap; font-family: monospace; font-size: 0.9em; "
        "background: #f8f9fa; padding: 1rem; border-radius: 0.4rem; overflow-x: auto;"
    )
    return f'<pre style="{pre_style}">' + "<br>".join(result) + "</pre>"
