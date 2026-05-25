"""Collect accuracy-latency benchmark data for RelayBP / MultiRelayBP with stop_nconv > 1.

With stop_nconv=1 the post-processing trick in accuracy_latency_tradeoff.ipynb lets a single
max-budget run represent the full accuracy-vs-iteration-budget curve. That trick breaks for
stop_nconv > 1 because the decoder may fall back to an earlier candidate if the budget is
tighter. Each (stop_nconv, num_relays) combination therefore needs its own benchmark run and
contributes one data point to the curve.

Results are written to the same CSV files read by the notebook, so adding entries with
stop_nconv > 1 to the `decoders` list in each notebook cell is all that is needed to plot
the new curves.

Usage
-----
    # Print the ordered task plan and exit (no benchmarking):
    uv run python benchmark-app/scripts/collect_accuracy_latency_data.py --dry-run

    # Run the full collection (leave running; Ctrl-C saves progress):
    uv run python benchmark-app/scripts/collect_accuracy_latency_data.py

    # Run with custom caps or worker count:
    uv run python benchmark-app/scripts/collect_accuracy_latency_data.py \\
        --errors-cap 200 --workers 4

For a remote server, detach with tmux or nohup:
    nohup uv run python benchmark-app/scripts/collect_accuracy_latency_data.py \\
        > collect_alt.log 2>&1 &
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

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


# --- Collection settings ---------------------------------------------------

STOP_NCONV_LIST = [2, 3, 4]

BATCH_SIZE = 128
ERRORS_CAP = 100
SHOTS_CAP = 300_000_000
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)

_COMMON_RELAY_PARAMS = {
    "max_iter_per_relay": 50,
    "pre_iter": 50,
}

# Each config entry specifies a circuit and the decoders to benchmark there.
# `max_num_relays` is the ceiling used by the notebook for this circuit.
# `num_relay_points` controls how many log-spaced relay counts to sample.
# For RSC_Phenom (only 9 relays) we enumerate all relay counts explicitly.
CONFIGS: list[dict[str, Any]] = [
    # {
    #     "circuit_name": "RotatedSurfaceCode_Phenom",
    #     "circuit_params": {"basis": "Z", "d": 11, "rounds": 11},
    #     "error_rate": 0.005,
    #     # All 9 relay counts — small enough to enumerate exhaustively.
    #     "relay_counts": list(range(1, 10)),
    #     "decoders": [
    #         (
    #             "RelayBP",
    #             {
    #                 "gamma0": 0.65,
    #                 "gamma_dist_interval": (-0.2241259276391633, 0.5608805426310253),
    #                 **_COMMON_RELAY_PARAMS,
    #             },
    #         ),
    #         (
    #             "MultiRelayBP",
    #             {
    #                 "gamma0": 0.65,
    #                 "gamma_dist_interval": (-0.2241259276391633, 0.5608805426310253),
    #                 "num_chains": 2,
    #                 **_COMMON_RELAY_PARAMS,
    #             },
    #         ),
    #         (
    #             "MultiRelayBP",
    #             {
    #                 "gamma0": 0.65,
    #                 "gamma_dist_interval": (-0.2241259276391633, 0.5608805426310253),
    #                 "num_chains": 3,
    #                 **_COMMON_RELAY_PARAMS,
    #             },
    #         ),
    #         (
    #             "MultiRelayBP",
    #             {
    #                 "gamma0": 0.65,
    #                 "gamma_dist_interval": (-0.2241259276391633, 0.5608805426310253),
    #                 "num_chains": 4,
    #                 **_COMMON_RELAY_PARAMS,
    #             },
    #         ),
    #     ],
    # },
    {
        "circuit_name": "BB_144_12_12_Circuit",
        "circuit_params": {"basis": "Z", "rounds": 12},
        "error_rate": 0.001,
        "relay_counts": sorted(
            set(np.round(np.geomspace(1, 65, 16) - 1).astype(int).tolist())
        ),
        "decoders": [
            (
                "RelayBP",
                {
                    "gamma0": 0.2,
                    "gamma_dist_interval": (-0.19806670996164882, 0.6200937872049607),
                    **_COMMON_RELAY_PARAMS,
                },
            ),
        ],
    },
    {
        "circuit_name": "RotatedSurfaceCode_Circuit",
        "circuit_params": {"basis": "Z", "d": 11, "rounds": 11},
        "error_rate": 0.001,
        "relay_counts": sorted(
            set(np.round(np.geomspace(1, 301, 20) - 1).astype(int).tolist())
        ),
        "decoders": [
            (
                "RelayBP",
                {
                    "gamma0": 0.55,
                    "gamma_dist_interval": (-0.2944735403909491, 0.9444054073490704),
                    **_COMMON_RELAY_PARAMS,
                },
            ),
            (
                "MultiRelayBP",
                {
                    "gamma0": 0.55,
                    "gamma_dist_interval": (-0.2944735403909491, 0.9444054073490704),
                    "num_chains": 2,
                    **_COMMON_RELAY_PARAMS,
                },
            ),
            (
                "MultiRelayBP",
                {
                    "gamma0": 0.55,
                    "gamma_dist_interval": (-0.2944735403909491, 0.9444054073490704),
                    "num_chains": 4,
                    **_COMMON_RELAY_PARAMS,
                },
            ),
        ],
    },
]


def build_tasks() -> list[tuple[TaskMetadata, Path]]:
    """Expand CONFIGS into a flat, cheap-first-ordered list of (metadata, csv_path).

    Each (stop_nconv, num_relays) pair yields one task.  Invalid combinations
    (num_relays < stop_nconv - 1) are silently skipped.

    Tasks are sorted by num_relays * stop_nconv so the cheapest runs finish first.
    """
    tasks: list[tuple[TaskMetadata, Path]] = []
    for config in CONFIGS:
        circuit_name = config["circuit_name"]
        circuit_params = config["circuit_params"]
        error_rate = config["error_rate"]
        relay_counts: list[int] = config["relay_counts"]

        for decoder_name, fixed_params in config["decoders"]:
            csv_path = get_csv_path(circuit_name, circuit_params, decoder_name)
            for stop_nconv in STOP_NCONV_LIST:
                for num_relays in relay_counts:
                    # Decoder constraint: stop_nconv <= num_relays + 1
                    if num_relays < stop_nconv - 1:
                        continue
                    decoder_params = {
                        **fixed_params,
                        "num_relays": num_relays,
                        "stop_nconv": stop_nconv,
                    }
                    metadata = TaskMetadata(
                        circuit_name=circuit_name,
                        circuit_params=circuit_params,
                        error_rate=error_rate,
                        decoder_name=decoder_name,
                        decoder_params=decoder_params,
                    )
                    tasks.append((metadata, csv_path))

    tasks.sort(
        key=lambda item: (
            item[0].decoder_params["num_relays"] * item[0].decoder_params["stop_nconv"]
        )
    )
    return tasks


def _task_summary(metadata: TaskMetadata) -> str:
    p = metadata.decoder_params
    chains = f" num_chains={p['num_chains']}" if "num_chains" in p else ""
    return (
        f"{metadata.circuit_name} {dict(metadata.circuit_params)} "
        f"| {metadata.decoder_name}{chains} p={metadata.error_rate:g} "
        f"num_relays={p['num_relays']} stop_nconv={p['stop_nconv']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered task plan and exit without benchmarking.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Parallel worker processes (0=serial). Default {NUM_WORKERS}.",
    )
    parser.add_argument(
        "--shots-cap",
        type=int,
        default=SHOTS_CAP,
        help=f"Per-task shots cap. Default {SHOTS_CAP}.",
    )
    parser.add_argument(
        "--errors-cap",
        type=int,
        default=ERRORS_CAP,
        help=f"Per-task observable-error target. Default {ERRORS_CAP}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Shots per batch. Default {BATCH_SIZE}.",
    )
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
        except Exception as exc:
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
