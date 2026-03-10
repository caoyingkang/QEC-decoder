"""Functions for benchmarking LearnedDMemBP decoder."""
from pathlib import Path

import numpy as np
import torch
import sinter
from qecdec import (
    RotatedSurfaceCode_Memory,
    DMemBPDecoder,
    SinterDecoderWrapper,
)

from utils import (
    BENCHMARK_CSV_FILENAME,
    is_consistent,
    extract_pytorch_decoder_name,
)


def _load_gamma_from_checkpoint(run_dir: Path) -> np.ndarray:
    ckpt_path = run_dir / "checkpoints" / "best_model.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["decoder.gamma"].numpy().astype(np.float64)


def run_LearnedDMemBP_benchmark(
    *,
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
    run_dir: Path,
    max_iter: int,
    p_list: list[float],
    max_shots: int,
    max_errors: int,
    num_workers: int,
):
    if not is_consistent(run_dir, code, noise_model, d, rounds, basis):
        raise ValueError(f"Run dir {run_dir} has inconsistent QEC config with the selected code, noise model, d, rounds, and basis.")

    csv_path = run_dir / BENCHMARK_CSV_FILENAME
    gamma = _load_gamma_from_checkpoint(run_dir)

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
        dec = DMemBPDecoder(
            expmt.chkmat, expmt.prior,
            gamma=gamma,
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
                    "decoder": extract_pytorch_decoder_name(run_dir),  # e.g., 'LearnedDMemBP/run_0'
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
