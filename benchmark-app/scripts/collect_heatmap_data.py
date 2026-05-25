"""Collect all Monte Carlo benchmark datapoints for the MultiRelayBP heatmaps.

This fills in every ``num_chains x num_relays`` grid cell that
``benchmark-app/notebooks/multirelaybp_heatmap.ipynb`` plots, for the three
circuit configurations defined there. Results are written to the same CSV files
the notebook reads (``benchmark-app/results/<circuit>/<params>/<decoder>/results.csv``),
so re-running the notebook afterwards picks up the new cells automatically.

Usage
-----
    # Print the ordered task plan and exit (no benchmarking):
    uv run python benchmark-app/scripts/collect_heatmap_data.py --dry-run

    # Run the full collection (default caps):
    uv run python benchmark-app/scripts/collect_heatmap_data.py

Each cell runs until it reaches ERRORS_CAP observable errors or SHOTS_CAP shots
(whichever comes first), mirroring the existing data. The total run is long
(~300 tasks, many needing ~1e8 shots); leave it running overnight. Completed
tasks are saved per-task and skipped on re-run, so the job resumes in place and
can span multiple sessions. For a remote server, detach with tmux or nohup:

    nohup uv run python benchmark-app/scripts/collect_heatmap_data.py > collect.log 2>&1 &
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make the benchmark-app package modules (constants, utils, learned_decoders)
# importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qecdec  # noqa: E402
from qecbench import CollectorParams, TaskMetadata, run_benchmark  # noqa: E402

from constants import CIRCUITS_ROOT  # noqa: E402
from utils import get_csv_path  # noqa: E402

# BB circuits load precompiled .stim files from this directory.
qecdec.circuits.BB_144_12_12_Circuit.load_dir = CIRCUITS_ROOT / "BB_144_12_12_Circuit"

# Register learned decoders (LearnedMultiRelayBP, ...) in the decoder registry.
import learned_decoders  # noqa: F401,E402


# --- Editable collection settings ------------------------------------------
LIST_NUM_CHAINS = [1, 2, 4, 8, 16]
LIST_NUM_RELAYS = [1, 2, 4, 8, 16]

BATCH_SIZE = 128
ERRORS_CAP = 100
SHOTS_CAP = 300_000_000  # safety cap; most cells stop on ERRORS_CAP first
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)  # 0 => serial

# Relay params shared by every config (and both decoders within a config).
_COMMON_RELAY_PARAMS = {
    "max_iter_per_relay": 50,
    "pre_iter": 50,
    "stop_nconv": 1,
}

# --- What to collect (mirrors the 3 notebook usage cells) ------------------
CONFIGS: list[dict[str, Any]] = [
    {
        "circuit_name": "RotatedSurfaceCode_Circuit",
        "circuit_params": {"basis": "Z", "d": 11, "rounds": 11},
        "error_rates": [0.001, 0.002, 0.003],
        "decoders": [
            (
                "MultiRelayBP",
                {
                    "gamma0": 0.55,
                    "gamma_dist_interval": (-0.2944735403909491, 0.9444054073490704),
                    **_COMMON_RELAY_PARAMS,
                },
            ),
        ],
    },
    {
        "circuit_name": "RotatedSurfaceCode_Phenom",
        "circuit_params": {"basis": "Z", "d": 11, "rounds": 11},
        "error_rates": [0.005, 0.006, 0.007],
        "decoders": [
            (
                "MultiRelayBP",
                {
                    "gamma0": 0.65,
                    "gamma_dist_interval": (-0.2241259276391633, 0.5608805426310253),
                    **_COMMON_RELAY_PARAMS,
                },
            ),
            (
                "LearnedMultiRelayBP",
                {
                    "ckpt_rel_path": "torchdecoder/runs/RotatedSurfaceCode_Phenom/basis=Z_d=11_rounds=11/LearnedDMemBP/run_0/checkpoints/best_model.ckpt",
                    "gamma_dist_interval": (-0.2241259276391633, 0.5608805426310253),
                    **_COMMON_RELAY_PARAMS,
                },
            ),
        ],
    },
    {
        "circuit_name": "BB_144_12_12_Circuit",
        "circuit_params": {"basis": "Z", "rounds": 12},
        "error_rates": [0.001, 0.002, 0.003],
        "decoders": [
            (
                "MultiRelayBP",
                {
                    "gamma0": 0.2,
                    "gamma_dist_interval": (-0.19806670996164882, 0.6200937872049607),
                    **_COMMON_RELAY_PARAMS,
                },
            ),
        ],
    },
]


def build_tasks() -> list[tuple[TaskMetadata, Path]]:
    """Expand CONFIGS into a flat, cheap-first-ordered list of (metadata, csv_path).

    Ordering by (num_chains * num_relays, -error_rate) front-loads the cells that
    finish fastest (smaller grids, higher error rates), so coverage accrues early.
    """
    tasks: list[tuple[TaskMetadata, Path]] = []
    for config in CONFIGS:
        circuit_name = config["circuit_name"]
        circuit_params = config["circuit_params"]
        for decoder_name, fixed_params in config["decoders"]:
            csv_path = get_csv_path(circuit_name, circuit_params, decoder_name)
            for error_rate in config["error_rates"]:
                for num_chains in LIST_NUM_CHAINS:
                    for num_relays in LIST_NUM_RELAYS:
                        decoder_params = {
                            **fixed_params,
                            "num_chains": num_chains,
                            "num_relays": num_relays,
                        }
                        metadata = TaskMetadata(
                            circuit_name=circuit_name,
                            circuit_params=circuit_params,
                            error_rate=error_rate,
                            decoder_name=decoder_name,
                            decoder_params=decoder_params,
                        )
                        tasks.append((metadata, csv_path))

    def cost_key(item: tuple[TaskMetadata, Path]) -> tuple[int, float]:
        params = item[0].decoder_params
        return (params["num_chains"] * params["num_relays"], -item[0].error_rate)

    tasks.sort(key=cost_key)
    return tasks


def _task_summary(metadata: TaskMetadata) -> str:
    p = metadata.decoder_params
    return (
        f"{metadata.circuit_name} {dict(metadata.circuit_params)} "
        f"| {metadata.decoder_name} p={metadata.error_rate:g} "
        f"num_chains={p['num_chains']} num_relays={p['num_relays']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered task plan and exit without benchmarking.",
    )
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Parallel worker processes (0=serial). Default {NUM_WORKERS}.")
    parser.add_argument("--shots-cap", type=int, default=SHOTS_CAP,
                        help=f"Per-cell shots cap. Default {SHOTS_CAP}.")
    parser.add_argument("--errors-cap", type=int, default=ERRORS_CAP,
                        help=f"Per-cell observable-error target. Default {ERRORS_CAP}.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Shots per batch. Default {BATCH_SIZE}.")
    args = parser.parse_args()

    tasks = build_tasks()
    num_tasks = len(tasks)

    if args.dry_run:
        print(f"Planned {num_tasks} benchmark tasks (cheap-first order):\n")
        for i, (metadata, csv_path) in enumerate(tasks, 1):
            print(f"[{i:3d}/{num_tasks}] {_task_summary(metadata)}")
            print(f"           -> {csv_path}")
        return

    collector_params = CollectorParams(
        batch_size=args.batch_size,
        shots_cap=args.shots_cap,
        errors_cap=args.errors_cap,
        num_parallel_workers=args.workers,
    )

    print(
        f"Collecting {num_tasks} tasks | errors_cap={args.errors_cap} "
        f"shots_cap={args.shots_cap:,} batch_size={args.batch_size} "
        f"workers={args.workers}\n"
    )

    t_start = time.time()
    completed = 0
    failed: list[str] = []
    for i, (metadata, csv_path) in enumerate(tasks, 1):
        print(f"\n========== [{i}/{num_tasks}] {_task_summary(metadata)} ==========")
        try:
            run_benchmark(
                metadata,
                collector_params,
                csv_path=csv_path,
                verbose=True,
            )
            completed += 1
        except KeyboardInterrupt:
            print("\nInterrupted. Completed tasks are saved; re-run to resume.")
            break
        except Exception as exc:  # keep going on a per-task failure
            print(f"!! Task failed: {exc!r}")
            failed.append(_task_summary(metadata))

    elapsed = time.time() - t_start
    print(
        f"\nDone. {completed}/{num_tasks} tasks ran this session "
        f"in {elapsed / 3600:.2f} h."
    )
    if failed:
        print(f"{len(failed)} task(s) failed:")
        for summary in failed:
            print(f"  - {summary}")


if __name__ == "__main__":
    main()
