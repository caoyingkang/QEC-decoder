"""Example: run a custom Monte Carlo benchmark headlessly.

Usage:

    uv run python benchmark-app/scripts/run_example.py

For long-running jobs on a remote server, run inside `tmux` or detach with
`nohup`, e.g.

    nohup uv run python benchmark-app/scripts/run_example.py > run.log 2>&1 &

Edit the parameters below to match your experiment. CSV outputs are appended
under ``benchmark-app/baselines-results/{code}_{noise_model}/...`` and existing
runs are resumed in place if you re-run after Ctrl-C.
"""

import os
from pathlib import Path

from qecbench import (
    BenchTaskParams,
    CollectorParams,
    QECParams,
    run_custom_benchmark,
)
from qecdec.experiments import RotatedSurfaceCode_Memory


qec_params = QECParams(
    code="RotatedSurfaceCode",
    noise_model="Phenomenological",
    d=5,
    rounds=10,
    basis="Z",
)

benchtask_params = BenchTaskParams(
    p_list=[0.001, 0.002, 0.003, 0.004, 0.005],
    baseline_decoder_params={
        "BP": {"max_iter": 50},
        "MWPM": {},
    },
    torchdecoder_shared_params={},
)

collector_params = CollectorParams(
    batch_size=128,
    shots_cap=200_000_000,
    errors_cap=100,
    device="cpu",
    num_parallel_workers=max(1, (os.cpu_count() or 1) - 1),
)

BASELINE_CSV_DIR = Path(__file__).resolve().parent.parent / "baselines-results"


if __name__ == "__main__":
    experiments = {
        p: RotatedSurfaceCode_Memory(
            d=qec_params.d,
            rounds=qec_params.rounds,
            basis=qec_params.basis,
            data_qubit_error_rate=p,
            meas_error_rate=p,
        )
        for p in benchtask_params.p_list
    }
    run_custom_benchmark(
        qec_params=qec_params,
        benchtask_params=benchtask_params,
        collector_params=collector_params,
        experiments=experiments,
        baseline_csv_dir=BASELINE_CSV_DIR,
        baseline_decoders=list(benchtask_params.baseline_decoder_params.keys()),
    )
