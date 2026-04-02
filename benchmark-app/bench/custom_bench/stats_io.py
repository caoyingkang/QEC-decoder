from pathlib import Path
from typing import Any

from ..params import BenchTaskParams, QECParams
from .collector_params import CollectorParams
from .stats import BenchmarkStats


def get_torchdecoder_csv_path(run_dir: Path) -> Path:
    return run_dir / "custom_benchmark.csv"


def get_baseline_csv_path(
    baseline_csv_dir: Path,
    qec_params: QECParams,
    decoder: str,
) -> Path:
    return baseline_csv_dir.joinpath(
        f"{qec_params.code}_{qec_params.noise_model}",
        f"d={qec_params.d}_rounds={qec_params.rounds}_basis={qec_params.basis}",
        f"{decoder}",
        "custom_benchmark.csv",
    )


def _filter_stats(
    stats: list[BenchmarkStats],
    *,
    p_list: list[float],
    decoder_params: dict[str, Any],
    collector_params: CollectorParams,
) -> list[BenchmarkStats]:
    """
    Filter the list of `BenchmarkStats` to only include those elements `s` such that:
    - `s.metadata.p` is in `p_list`
    - `s.metadata.decoder_params` equals `decoder_params`
    - `s.is_complete(collector_params.shots_cap, collector_params.errors_cap)` equals `True`
    """
    filtered: list[BenchmarkStats] = []
    for s in stats:
        if s.metadata.p not in p_list:
            continue
        if s.metadata.decoder_params != decoder_params:
            continue
        if not s.is_complete(collector_params.shots_cap, collector_params.errors_cap):
            continue
        filtered.append(s)
    return filtered


def load_and_merge_stats(
    run_dirs: list[Path],
    baseline_decoders: list[str],
    baseline_csv_dir: Path,
    *,
    benchtask_params: BenchTaskParams,
    collector_params: CollectorParams,
    qec_params: QECParams,
) -> tuple[list[BenchmarkStats], list[Path], list[str]]:
    """
    Load and merge PyTorch decoders' and baseline decoders' stats into a single list.

    Return `(all_stats, pending_run_dirs, pending_baseline_decoders)` where:
    - `all_stats` is a merged list of all `BenchmarkStats` consistent with the given parameters.
    - `pending_run_dirs` is a sublist of `run_dirs` with missing or incomplete data.
    - `pending_baseline_decoders` is a sublist of `baseline_decoders` with missing or incomplete data.
    """
    all_stats: list[BenchmarkStats] = []
    pending_run_dirs: list[Path] = []
    pending_baseline_decoders: list[str] = []

    # Load PyTorch decoders' stats.
    for run_dir in run_dirs:
        csv_path = get_torchdecoder_csv_path(run_dir)
        if not csv_path.exists():
            pending_run_dirs.append(run_dir)
            continue
        stats = BenchmarkStats.load_csv(csv_path)
        stats = _filter_stats(
            stats,
            p_list=benchtask_params.p_list,
            decoder_params=benchtask_params.torchdecoder_shared_params,
            collector_params=collector_params,
        )
        if set(s.metadata.p for s in stats) != set(benchtask_params.p_list):
            pending_run_dirs.append(run_dir)
            continue
        all_stats.extend(stats)

    # Load baseline decoders' stats.
    for decoder in baseline_decoders:
        csv_path = get_baseline_csv_path(baseline_csv_dir, qec_params, decoder)
        if not csv_path.exists():
            pending_baseline_decoders.append(decoder)
            continue
        stats = BenchmarkStats.load_csv(csv_path)
        stats = _filter_stats(
            stats,
            p_list=benchtask_params.p_list,
            decoder_params=benchtask_params.baseline_decoder_params[decoder],
            collector_params=collector_params,
        )
        if set(s.metadata.p for s in stats) != set(benchtask_params.p_list):
            pending_baseline_decoders.append(decoder)
            continue
        all_stats.extend(stats)

    return all_stats, pending_run_dirs, pending_baseline_decoders
