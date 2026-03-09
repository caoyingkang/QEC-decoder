"""Functions for benchmarking baseline decoders."""
from pathlib import Path

import sinter
from qecdec import (
    RotatedSurfaceCode_Memory,
    BPDecoder,
    SinterDecoderWrapper,
)

from utils import BENCHMARK_ROOT

BASELINES_DIR = BENCHMARK_ROOT / "baselines"
BASELINE_DECODERS = ["MWPM", "BP"]


def get_baseline_csv_path(
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
    decoder: str,
) -> Path:
    return BASELINES_DIR.joinpath(
        f"{code}_{noise_model}",
        f"d={d}_rounds={rounds}_basis={basis}",
        f"{decoder}.csv",
    )


def run_MWPM_benchmark(
    *,
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
    p_list: list[float],
    max_shots: int,
    max_errors: int,
    num_workers: int,
):
    csv_path = get_baseline_csv_path(
        code, noise_model, d, rounds, basis, "MWPM"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[sinter.Task] = []
    for p in p_list:
        expmt = RotatedSurfaceCode_Memory(
            d=d,
            rounds=rounds,
            basis=basis,
            data_qubit_error_rate=p,
            meas_error_rate=p,
        )
        tasks.append(
            sinter.Task(
                circuit=expmt.circuit,
                detector_error_model=expmt.dem,
                decoder="pymatching",  # Built-in MWPM decoder in sinter
                json_metadata={
                    "d": d,
                    "rounds": rounds,
                    "basis": basis,
                    "p": p,
                    "decoder": "MWPM",
                },
            )
        )

    sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        save_resume_filepath=csv_path,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )


def run_BP_benchmark(
    *,
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
    max_iter: int,
    p_list: list[float],
    max_shots: int,
    max_errors: int,
    num_workers: int,
):
    csv_path = get_baseline_csv_path(
        code, noise_model, d, rounds, basis, "BP"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[sinter.Task] = []
    custom_decoders: dict[str, sinter.Decoder] = {}
    for p in p_list:
        expmt = RotatedSurfaceCode_Memory(
            d=d,
            rounds=rounds,
            basis=basis,
            data_qubit_error_rate=p,
            meas_error_rate=p,
        )
        dec = BPDecoder(
            expmt.chkmat, expmt.prior,
            max_iter=max_iter,
        )
        custom_decoder_id = f"custom_decoder_{len(custom_decoders)}"
        custom_decoders[custom_decoder_id] = SinterDecoderWrapper(dec, expmt.obsmat)
        tasks.append(
            sinter.Task(
                circuit=expmt.circuit,
                detector_error_model=expmt.dem,
                decoder=custom_decoder_id,
                json_metadata={
                    "d": d,
                    "rounds": rounds,
                    "basis": basis,
                    "p": p,
                    "decoder": "BP",
                    "max_iter": max_iter,
                },
            )
        )

    sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        save_resume_filepath=csv_path,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
        custom_decoders=custom_decoders,
    )
