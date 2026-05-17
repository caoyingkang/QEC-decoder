"""CSV path conventions and bulk stats loading."""

from pathlib import Path
from typing import Any, NamedTuple

from .params import BenchTaskParams, CollectorParams, QECParams
from .stats import BenchmarkStats


def get_torchdecoder_csv_path(run_dir: Path) -> Path:
    """Path where a Lightning run's custom-benchmark CSV is stored."""
    return run_dir / "custom_benchmark.csv"


def get_baseline_csv_path(
    baseline_csv_dir: Path,
    qec_params: QECParams,
    decoder: str,
) -> Path:
    """Path where the CSV for one ``(qec_params, decoder)`` baseline lives.

    Layout:
    ``{baseline_csv_dir}/{code}_{noise_model}/d={d}_rounds={rounds}_basis={basis}/{decoder}/custom_benchmark.csv``
    """
    return baseline_csv_dir.joinpath(
        f"{qec_params.code}_{qec_params.noise_model}",
        f"d={qec_params.d}_rounds={qec_params.rounds}_basis={qec_params.basis}",
        f"{decoder}",
        "custom_benchmark.csv",
    )


class StatsSource(NamedTuple):
    """A single CSV to load benchmark stats from.

    Attributes
    ----------
    display_name : str
        Name reported back in the ``pending_display_names`` list if this
        source is missing or incomplete.
    csv_path : Path
        Path to the result CSV (may or may not exist yet).
    expected_decoder_params : dict
        ``decoder_params`` value that loaded ``BenchmarkStats`` must match
        for them to be considered "current" (older runs with different
        params are ignored, which causes this source to be reported as
        pending).
    """

    display_name: str
    csv_path: Path
    expected_decoder_params: dict[str, Any]


def _filter_stats(
    stats: list[BenchmarkStats],
    *,
    p_list: list[float],
    decoder_params: dict[str, Any],
    shots_cap: int,
    errors_cap: int,
) -> list[BenchmarkStats]:
    """Keep stats whose metadata matches and which are complete."""
    return [
        s
        for s in stats
        if s.metadata.p in p_list
        and s.metadata.decoder_params == decoder_params
        and s.is_complete(shots_cap, errors_cap)
    ]


def load_and_filter_stats(
    sources: list[StatsSource],
    *,
    p_list: list[float],
    collector_params: CollectorParams,
) -> tuple[list[BenchmarkStats], list[str]]:
    """Load stats from each source, filtering to complete entries matching ``p_list``.

    A source is considered "pending" (needing more data) if its CSV is
    missing, or if after filtering by ``expected_decoder_params`` and
    completion it does not cover every ``p`` in ``p_list``.

    Returns
    -------
    all_stats : list[BenchmarkStats]
        Flat list of all loaded, current, complete stats across sources.
    pending_display_names : list[str]
        Names of sources that still have missing or incomplete coverage.
    """
    all_stats: list[BenchmarkStats] = []
    pending: list[str] = []
    p_set = set(p_list)

    for source in sources:
        if not source.csv_path.exists():
            pending.append(source.display_name)
            continue
        stats = _filter_stats(
            BenchmarkStats.load_csv(source.csv_path),
            p_list=p_list,
            decoder_params=source.expected_decoder_params,
            shots_cap=collector_params.shots_cap,
            errors_cap=collector_params.errors_cap,
        )
        if {s.metadata.p for s in stats} != p_set:
            pending.append(source.display_name)
            continue
        all_stats.extend(stats)

    return all_stats, pending


def build_baseline_sources(
    baseline_csv_dir: Path,
    *,
    qec_params: QECParams,
    benchtask_params: BenchTaskParams,
    baseline_decoders: list[str],
) -> list[StatsSource]:
    """Convenience: build :class:`StatsSource` entries for baseline decoders.

    Uses :func:`get_baseline_csv_path` to resolve the CSV path and looks up
    ``expected_decoder_params`` from
    ``benchtask_params.baseline_decoder_params``.
    """
    return [
        StatsSource(
            display_name=decoder,
            csv_path=get_baseline_csv_path(baseline_csv_dir, qec_params, decoder),
            expected_decoder_params=benchtask_params.baseline_decoder_params[decoder],
        )
        for decoder in baseline_decoders
    ]
