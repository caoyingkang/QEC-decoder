"""Main functions for benchmarking."""
from pathlib import Path

from baselines_benchmark import (
    run_MWPM_benchmark,
    run_BP_benchmark,
)
from pytorch_decoder_benchmark import run_pytorch_decoder_benchmark


def run_benchmark(
    *,
    code: str,
    noise_model: str,
    d: int,
    rounds: int,
    basis: str,
    run_dirs: list[Path],
    baseline_decoders: list[str],
    max_iter: int,
    p_list: list[float],
    max_shots: int,
    max_errors: int,
    num_workers: int,
    device: str,
    bypass: bool,
    use_prior_in_ckpt: bool,
):
    # Benchmark PyTorch decoders
    for run_dir in run_dirs:
        run_pytorch_decoder_benchmark(
            code=code,
            noise_model=noise_model,
            d=d,
            rounds=rounds,
            basis=basis,
            run_dir=run_dir,
            max_iter=max_iter,
            p_list=p_list,
            max_shots=max_shots,
            max_errors=max_errors,
            num_workers=num_workers,
            device=device,
            bypass=bypass,
            use_prior_in_ckpt=use_prior_in_ckpt,
        )

    # Benchmark baseline decoders
    if "MWPM" in baseline_decoders:
        run_MWPM_benchmark(
            code=code,
            noise_model=noise_model,
            d=d,
            rounds=rounds,
            basis=basis,
            p_list=p_list,
            max_shots=max_shots,
            max_errors=max_errors,
            num_workers=num_workers,
        )
    if "BP" in baseline_decoders:
        run_BP_benchmark(
            code=code,
            noise_model=noise_model,
            d=d,
            rounds=rounds,
            basis=basis,
            max_iter=max_iter,
            p_list=p_list,
            max_shots=max_shots,
            max_errors=max_errors,
            num_workers=num_workers,
        )
