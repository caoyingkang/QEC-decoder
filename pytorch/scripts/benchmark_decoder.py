"""
Monte Carlo benchmarking of trained decoders.

Baselines (MWPM, BP) are cached per code config in baselines/; 
learned decoder results (DMemBP) are cached per run in runs/.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Protocol
from collections import defaultdict

import numpy as np
import sinter
import torch
from omegaconf import OmegaConf, DictConfig
from qecdec import BPDecoder, DMemBPDecoder, RotatedSurfaceCode_Memory
from qecdec.sinter_wrapper import SinterDecoderWrapper

_PYTORCH_ROOT = Path(__file__).resolve().parent.parent
_RUNS_ROOT = _PYTORCH_ROOT / "runs"
_BASELINES_ROOT = _PYTORCH_ROOT / "baselines"

DEFAULT_P_LIST = [0.004, 0.006, 0.008, 0.01, 0.012]
DEFAULT_MAX_SHOTS = 1_000_000
DEFAULT_MAX_ERRORS = 100
DEFAULT_MAX_ITER = 50
BENCHMARK_RESULTS_FILENAME = "benchmark_results.csv"


def get_baselines_path(d: int, rounds: int, basis: str) -> Path:
    """Path to benchmark_results.csv for baseline decoders."""
    qec_expmt = f"d={d}_rounds={rounds}_basis={basis}"
    return _BASELINES_ROOT / "rotated_surface_code_memory" / qec_expmt / BENCHMARK_RESULTS_FILENAME


def discover_runs() -> list[Path]:
    """Find all run_* directories that contain checkpoints/best_model.ckpt."""
    runs = []
    for p in _RUNS_ROOT.rglob("checkpoints/best_model.ckpt"):
        run_dir = p.parent.parent
        if run_dir.name.startswith("run_"):
            runs.append(run_dir)
    return sorted(runs)


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


def baseline_tasks_complete(baselines_path: Path, p_list: list[float]) -> bool:
    """Check if baselines CSV has MWPM and BP data (by decoder in json_metadata)."""
    # TODO: check if MWPM and BP data are present for all p in p_list
    if not baselines_path.exists():
        return False
    with open(baselines_path) as f:
        content = f.read()
    # json_metadata contains "decoder": "pymatching" or "decoder": "bp"
    norm = content.replace(" ", "")
    return "pymatching" in content and ('"decoder":"bp"' in norm or '"decoder":"bp"}' in norm)


def run_baseline_benchmark(
    d: int,
    rounds: int,
    basis: str,
    p_list: list[float],
    max_iter: int,
    max_shots: int,
    max_errors: int,
    num_workers: int,
):
    """Run MWPM + BP benchmark, cache to baselines/.../benchmark_results.csv."""
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
        # MWPM decoder
        tasks.append(sinter.Task(
            circuit=expmt.circuit,
            detector_error_model=expmt.dem,
            decoder="pymatching",
            json_metadata={"d": d, "rounds": rounds, "basis": basis, "p": p, "decoder": "pymatching"},
        ))
        # Vanilla BP decoder
        bp = BPDecoder(expmt.chkmat, expmt.prior, max_iter=max_iter)
        bp_id = f"bp_{len(custom_decoders)}"
        custom_decoders[bp_id] = SinterDecoderWrapper(bp, expmt.obsmat)
        tasks.append(sinter.Task(
            circuit=expmt.circuit,
            detector_error_model=expmt.dem,
            decoder=bp_id,
            json_metadata={"d": d, "rounds": rounds, "basis": basis, "p": p, "decoder": "bp"},
        ))

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
):
    """Run learned DMemBP benchmark for one run, cache to run_dir/benchmark_results.csv."""
    results_path = run_dir / BENCHMARK_RESULTS_FILENAME
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
            expmt.chkmat, expmt.prior,
            gamma=gamma, max_iter=max_iter,
        )
        dmembp_id = f"learned_dmembp_{len(custom_decoders)}"
        custom_decoders[dmembp_id] = SinterDecoderWrapper(dmembp, expmt.obsmat)
        tasks.append(sinter.Task(
            circuit=expmt.circuit,
            detector_error_model=expmt.dem,
            decoder=dmembp_id,
            json_metadata={"d": d, "rounds": rounds, "basis": basis, "p": p, "decoder": "learned_dmembp"},
        ))

    sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        custom_decoders=custom_decoders,
        save_resume_filepath=results_path,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )


class CLIArgs(Protocol):
    runs: list[Path]
    all: bool
    p_list: list[float]
    max_iter: int
    max_shots: int
    max_errors: int
    num_workers: int


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo benchmark of trained decoders")
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Run directories to benchmark (e.g. runs/.../learned_dmembp/run_0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all discovered runs",
    )
    parser.add_argument(
        "--p-list",
        type=float,
        nargs="+",
        default=DEFAULT_P_LIST,
        help="Physical error rates to benchmark at",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Maximum number of iterations for BP decoder (or its relatives, e.g. DMemBP)",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=DEFAULT_MAX_SHOTS,
        help="Stop Monte Carlo benchmarking after this many shots",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help="Stop Monte Carlo benchmarking after this many decoding failures",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of workers to use for Monte Carlo benchmarking (default: number of cpu cores - 1)",
    )
    args: CLIArgs = parser.parse_args()

    if args.all:
        run_dirs = discover_runs()
        if len(run_dirs) == 0:
            print("No runs found.")
            return
    elif args.runs:
        run_dirs = []
        for r in args.runs:
            r = Path(r).resolve()
            if not r.exists():
                print(f"Warning: {r} does not exist, skipping")
                continue
            if not (r / "checkpoints" / "best_model.ckpt").exists():
                print(f"Warning: {r} has no best_model.ckpt, skipping")
                continue
            run_dirs.append(r)
    else:
        parser.error("Provide run directories or --all")

    # Group runs by qec experiment config
    by_qec_expmts = defaultdict[tuple[int, int, str], list[Path]](list)
    for run_dir in run_dirs:
        cfg = load_run_config(run_dir)
        qec_cfg = cfg.qec
        d = qec_cfg.d
        rounds = qec_cfg.rounds
        basis = qec_cfg.basis
        by_qec_expmts[(d, rounds, basis)].append(run_dir)

    for (d, rounds, basis), runs in by_qec_expmts.items():
        baselines_path = get_baselines_path(d, rounds, basis)
        if not baseline_tasks_complete(baselines_path, args.p_list):
            print(f">>>>>> Running baselines (MWPM, BP) for d={d} rounds={rounds} basis={basis}")
            run_baseline_benchmark(
                d=d, rounds=rounds, basis=basis,
                p_list=args.p_list,
                max_iter=args.max_iter,
                max_shots=args.max_shots,
                max_errors=args.max_errors,
                num_workers=args.num_workers,
            )
        else:
            print(f">>>>>> Baselines already cached for d={d} rounds={rounds} basis={basis}")

        for run_dir in runs:
            print(f">>>>>> Benchmarking learned DMemBP: {run_dir}")
            run_learned_dmembp_benchmark(
                run_dir=run_dir,
                d=d, rounds=rounds, basis=basis,
                p_list=args.p_list,
                max_iter=args.max_iter,
                max_shots=args.max_shots,
                max_errors=args.max_errors,
                num_workers=args.num_workers,
            )


if __name__ == "__main__":
    sys.path.insert(0, str(_PYTORCH_ROOT))
    main()
