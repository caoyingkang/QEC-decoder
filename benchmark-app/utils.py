from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qecbench import TaskMetadata, TaskStats

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


def check_benchmark_completeness(
    task_metadata: TaskMetadata,
    csv_path: Path,
    shots_cap: int,
    errors_cap: int,
) -> bool:
    stats_list = TaskStats.load_csv(csv_path)
    stats = TaskStats.find_by_metadata(stats_list, task_metadata)
    return stats is not None and stats.is_complete(shots_cap, errors_cap)
