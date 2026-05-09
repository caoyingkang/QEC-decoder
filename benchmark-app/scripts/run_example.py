"""Example: run a custom Monte Carlo benchmark headlessly.

Usage:

    uv run python benchmark-app/scripts/run_example.py

For long-running jobs on a remote server, run inside `tmux` or detach with
`nohup`, e.g.

    nohup uv run python benchmark-app/scripts/run_example.py > run.log 2>&1 &

Edit the parameters below to match your experiment. CSV outputs are appended
under `benchmark-app/baselines-results/{code}_{noise_model}/...` and existing
runs are resumed in place if you re-run after Ctrl-C.
"""

import os
import sys
from pathlib import Path

# Make the top-level `bench`, `constants`, `experiment_factory`, and
# `torchdecoder_utils` modules importable. These live at `benchmark-app/`
# (not inside an installable package), so we prepend it to `sys.path`.
# This makes the script work from any CWD; copy this preamble when you
# write your own runner script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.custom_bench.collector_params import CollectorParams  # noqa: E402
from bench.custom_bench.run import run_custom_benchmark  # noqa: E402
from bench.params import BenchTaskParams, QECParams  # noqa: E402


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


if __name__ == "__main__":
    run_custom_benchmark(
        qec_params=qec_params,
        benchtask_params=benchtask_params,
        collector_params=collector_params,
        baseline_decoders=list(benchtask_params.baseline_decoder_params.keys()),
        torchdecoder_run_dirs=[],
    )
