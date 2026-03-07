"""
Monte Carlo benchmarking of trained decoders.

Baselines (MWPM, BP) are cached per code config in baselines/;
learned decoder results (DMemBP) are cached per run in runs/.
"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import sinter
import torch
from omegaconf import OmegaConf, DictConfig
from qecdec import BPDecoder, DMemBPDecoder, RotatedSurfaceCode_Memory
from qecdec.sinter_wrapper import SinterDecoderWrapper

PYTORCH_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PYTORCH_ROOT / "runs"
BASELINES_ROOT = PYTORCH_ROOT / "baselines"

BENCHMARK_RESULTS_FILENAME = "benchmark_results.csv"
DEFAULT_MAX_ITER = 50


def get_baselines_path(d: int, rounds: int, basis: str) -> Path:
    """Path to benchmark_results.csv for baseline decoders."""
    qec_expmt = f"d={d}_rounds={rounds}_basis={basis}"
    return BASELINES_ROOT / "rotated_surface_code_memory" / qec_expmt / BENCHMARK_RESULTS_FILENAME


def load_run_config(run_dir: Path) -> DictConfig:
    """Load config.yaml from a run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return OmegaConf.load(config_path)


def load_gamma_from_checkpoint(run_dir: Path) -> np.ndarray:
    """Load gamma from best_model.ckpt."""
    ckpt_path = run_dir / "checkpoints" / "best_model.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["decoder.gamma"].numpy().astype(np.float64)


def baseline_tasks_complete(
    baselines_path: Path, p_list: list[float], baseline_decoders: list[str]
) -> bool:
    """Check if baselines CSV has data for all requested decoders (by decoder in json_metadata)."""
    if not baselines_path.exists():
        return False
    with open(baselines_path) as f:
        content = f.read()
    norm = content.replace(" ", "")
    for decoder in baseline_decoders:
        if decoder == "bp":
            if '"decoder":"bp"' not in norm and '"decoder":"bp"}' not in norm:
                return False
        elif decoder == "pymatching":
            if "pymatching" not in content:
                return False
        else:
            return False
    return True


def run_baseline_benchmark(
    d: int,
    rounds: int,
    basis: str,
    p_list: list[float],
    max_iter: int,
    max_shots: int,
    max_errors: int,
    num_workers: int,
    baseline_decoders: list[str],
) -> None:
    """Run selected baseline decoders, cache to baselines/.../benchmark_results.csv."""
    if len(baseline_decoders) == 0:
        return

    baselines_path = get_baselines_path(d, rounds, basis)
    baselines_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    custom_decoders = {}
    for p in p_list:
        expmt = RotatedSurfaceCode_Memory(
            d=d,
            rounds=rounds,
            basis=basis,
            data_qubit_error_rate=p,
            meas_error_rate=p,
        )
        if "pymatching" in baseline_decoders:
            tasks.append(
                sinter.Task(
                    circuit=expmt.circuit,
                    detector_error_model=expmt.dem,
                    decoder="pymatching",
                    json_metadata={
                        "d": d,
                        "rounds": rounds,
                        "basis": basis,
                        "p": p,
                        "decoder": "pymatching",
                    },
                )
            )
        if "bp" in baseline_decoders:
            bp = BPDecoder(expmt.chkmat, expmt.prior, max_iter=max_iter)
            bp_id = f"bp_{len(custom_decoders)}"
            custom_decoders[bp_id] = SinterDecoderWrapper(bp, expmt.obsmat)
            tasks.append(
                sinter.Task(
                    circuit=expmt.circuit,
                    detector_error_model=expmt.dem,
                    decoder=bp_id,
                    json_metadata={
                        "d": d,
                        "rounds": rounds,
                        "basis": basis,
                        "p": p,
                        "decoder": "bp",
                    },
                )
            )

    sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        custom_decoders=custom_decoders,
        save_resume_filepath=baselines_path,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )


def run_learned_dmembp_benchmark(
    run_dir: Path,
    d: int,
    rounds: int,
    basis: str,
    p_list: list[float],
    max_iter: int,
    max_shots: int,
    max_errors: int,
    num_workers: int,
) -> None:
    """Run learned DMemBP benchmark for one run, cache to run_dir/benchmark_results.csv."""
    gamma = load_gamma_from_checkpoint(run_dir)

    tasks = []
    custom_decoders = {}
    for p in p_list:
        expmt = RotatedSurfaceCode_Memory(
            d=d,
            rounds=rounds,
            basis=basis,
            data_qubit_error_rate=p,
            meas_error_rate=p,
        )
        dmembp = DMemBPDecoder(
            expmt.chkmat,
            expmt.prior,
            gamma=gamma,
            max_iter=max_iter,
        )
        dmembp_id = f"learned_dmembp_{len(custom_decoders)}"
        custom_decoders[dmembp_id] = SinterDecoderWrapper(dmembp, expmt.obsmat)
        tasks.append(
            sinter.Task(
                circuit=expmt.circuit,
                detector_error_model=expmt.dem,
                decoder=dmembp_id,
                json_metadata={
                    "d": d,
                    "rounds": rounds,
                    "basis": basis,
                    "p": p,
                    "decoder": "learned_dmembp",
                },
            )
        )

    results_path = run_dir / BENCHMARK_RESULTS_FILENAME
    sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        custom_decoders=custom_decoders,
        save_resume_filepath=results_path,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )


def run_benchmark(
    run_dirs: list[Path],
    p_list: list[float],
    max_shots: int,
    max_errors: int,
    baselines: list[str],
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    num_workers: int | None = None,
) -> None:
    """
    Run baseline and learned DMemBP benchmarks for the given run directories.

    Baselines are cached per QEC config; learned results are cached per run.
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    # Group runs by QEC experiment config
    by_qec_expmts: dict[tuple[int, int, str], list[Path]] = defaultdict(list)
    for run_dir in run_dirs:
        cfg = load_run_config(run_dir)
        qec_cfg = cfg.qec
        d = qec_cfg.d
        rounds = qec_cfg.rounds
        basis = qec_cfg.basis
        by_qec_expmts[(d, rounds, basis)].append(run_dir)

    for (d, rounds, basis), runs in by_qec_expmts.items():
        baselines_path = get_baselines_path(d, rounds, basis)
        if not baseline_tasks_complete(baselines_path, p_list, baselines):
            run_baseline_benchmark(
                d=d,
                rounds=rounds,
                basis=basis,
                p_list=p_list,
                max_iter=max_iter,
                max_shots=max_shots,
                max_errors=max_errors,
                num_workers=num_workers,
                baseline_decoders=baselines,
            )

        for run_dir in runs:
            run_learned_dmembp_benchmark(
                run_dir=run_dir,
                d=d,
                rounds=rounds,
                basis=basis,
                p_list=p_list,
                max_iter=max_iter,
                max_shots=max_shots,
                max_errors=max_errors,
                num_workers=num_workers,
            )
