from collections.abc import Mapping
from pathlib import Path
from typing import Any

from constants import RESULTS_DIR


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
