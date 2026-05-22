"""Constants and default decoder hyperparameters for the benchmark app."""

import json
from pathlib import Path
from typing import Any


# --- Path constants ----------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_APP_ROOT = REPO_ROOT / "benchmark-app"
RESULTS_DIR = BENCHMARK_APP_ROOT / "results"
TORCH_RUNS_ROOT = REPO_ROOT / "torchdecoder" / "runs"
CIRCUITS_ROOT = REPO_ROOT / "circuits"

# --- Monte Carlo collector parameters ----------------------------
DEFAULT_BATCH_SIZE = 128
DEFAULT_SHOTS_CAP = 100_000_000
DEFAULT_ERRORS_CAP = 100

# --- Circuit parameters ------------------------------------------
DEFAULT_ERROR_RATE = 0.001


# --- Decoder parameters ------------------------------------------
def _load_default_decoder_params_json_dict() -> dict[str, Any]:
    json_path = BENCHMARK_APP_ROOT / "default_decoder_params.json"
    with open(json_path, "r") as f:
        params = json.load(f)
    return params


_DEFAULT_DECODER_PARAMS = _load_default_decoder_params_json_dict()


def get_default_decoder_params(
    circuit_name: str, circuit_params_str: str, decoder_name: str
) -> dict[str, Any]:
    return _DEFAULT_DECODER_PARAMS[circuit_name][circuit_params_str][decoder_name]
