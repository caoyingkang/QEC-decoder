"""Unified benchmark for PyTorch decoders (LearnedDMemBP, MultiDMemBP, etc.)."""
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import sinter
from qecdec import RotatedSurfaceCode_Memory

from utils import (
    PYTORCH_ROOT,
    BENCHMARK_CSV_FILENAME,
    is_consistent,
    extract_pytorch_decoder_name,
)

# Add pytorch/ to sys.path for src imports
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from src.models import DecoderModel, build_decoder_model
from src.sinter_interface import PyTorchSinterDecoder


def _load_decoder_model_from_run_dir(
    chkmat: np.ndarray,
    prior: np.ndarray,
    run_dir: Path,
    max_iter: int,
    *,
    use_prior_in_ckpt: bool,
) -> DecoderModel:
    """
    Load a `DecoderModel` from a training run directory.

    Expect the directory `run_dir` to contain `config.yaml` and `checkpoints/best_model.ckpt`.
    Use `max_iter` as the number of iterations for inference (overriding config.yaml).

    If `use_prior_in_ckpt` is True, load and use the prior LLRs from the checkpoint. Otherwise, 
    use the prior passed as an argument.
    """
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    ckpt_path = run_dir / "checkpoints" / "best_model.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model
    model_cfg.num_iters = max_iter

    model = build_decoder_model(chkmat, prior, model_cfg)
    if use_prior_in_ckpt:
        model.load_lightning_checkpoint(ckpt_path, skip_keys=[])
    else:
        model.load_lightning_checkpoint(ckpt_path, skip_keys=["prior_llr"])
    return model


def run_pytorch_decoder_benchmark(
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
    device: str,
    bypass: bool,
    use_prior_in_ckpt: bool,
):
    if not is_consistent(run_dir, code, noise_model, d, rounds, basis):
        raise ValueError(
            f"Run dir {run_dir} has inconsistent QEC config with the selected "
            f"code, noise_model, d, rounds, and basis."
        )
    csv_path = run_dir / BENCHMARK_CSV_FILENAME

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
        model = _load_decoder_model_from_run_dir(
            expmt.chkmat,
            expmt.prior,
            run_dir,
            max_iter,
            use_prior_in_ckpt=use_prior_in_ckpt,
        )
        custom_decoder_id = f"custom_decoder_{len(custom_decoders)}"
        custom_decoders[custom_decoder_id] = PyTorchSinterDecoder(
            model,
            expmt.obsmat,
            device=device,
            bypass=bypass,
        )
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
                    "decoder": extract_pytorch_decoder_name(run_dir),
                    "max_iter": max_iter,
                    "bypass": bypass,
                    "use_prior_in_ckpt": use_prior_in_ckpt,
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
