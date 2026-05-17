"""Monte Carlo benchmarking for QEC decoders.

qecbench bundles qecdec and torchdecoder-core decoders into a unified
:class:`BenchmarkDecoder` interface, drives them through a parallel-capable
Monte Carlo collector, and persists results as CSV-backed
:class:`BenchmarkStats`. Plotting helpers return matplotlib / plotly figures
ready to render anywhere.

The package is UI-agnostic and never imports a hardcoded data path; callers
supply pre-built :class:`qecdec.experiments.Experiment` objects and explicit
checkpoint / CSV paths.
"""

from .constants import (
    ALL_BASELINE_DECODERS,
    BASELINE_DECODERS_GRAPHLIKE,
    BASELINE_DECODERS_HYPERGRAPH,
    DEFAULT_BASELINE_DECODERS_GRAPHLIKE,
    DEFAULT_BASELINE_DECODERS_HYPERGRAPH,
)
from .collector import collect_stats
from .decoders import (
    BenchmarkDecoder,
    DecodeResult,
    PyTorchBenchmarkDecoder,
    QecdecBenchmarkDecoder,
)
from .io import (
    StatsSource,
    build_baseline_sources,
    get_baseline_csv_path,
    get_torchdecoder_csv_path,
    load_and_filter_stats,
)
from .params import (
    BenchTaskParams,
    CollectorParams,
    QECParams,
    TorchDecoderTask,
)
from .plotting import (
    plot_avg_iters_vs_per,
    plot_fr_vs_per,
    plot_iters_distribution,
    plot_ler_vs_per,
    plot_smr_vs_per,
)
from .runner import (
    run_baseline_benchmark,
    run_custom_benchmark,
    run_torchdecoder_benchmark,
)
from .stats import BenchmarkStats, TaskMetadata
from .torch_loader import (
    load_gamma_from_checkpoint,
    load_prior_from_checkpoint,
    load_torchdecoder,
)
from .types import Bit2DArray, Bool1DArray, Float1DArray, Int1DArray

__all__ = [
    # constants
    "ALL_BASELINE_DECODERS",
    "BASELINE_DECODERS_GRAPHLIKE",
    "BASELINE_DECODERS_HYPERGRAPH",
    "DEFAULT_BASELINE_DECODERS_GRAPHLIKE",
    "DEFAULT_BASELINE_DECODERS_HYPERGRAPH",
    # parameters
    "BenchTaskParams",
    "CollectorParams",
    "QECParams",
    "TorchDecoderTask",
    "Bit2DArray",
    "Bool1DArray",
    "Float1DArray",
    "Int1DArray",
    # stats
    "BenchmarkStats",
    "TaskMetadata",
    # decoders
    "BenchmarkDecoder",
    "DecodeResult",
    "PyTorchBenchmarkDecoder",
    "QecdecBenchmarkDecoder",
    # collection + orchestration
    "collect_stats",
    "run_baseline_benchmark",
    "run_custom_benchmark",
    "run_torchdecoder_benchmark",
    # IO
    "StatsSource",
    "build_baseline_sources",
    "get_baseline_csv_path",
    "get_torchdecoder_csv_path",
    "load_and_filter_stats",
    # torch loaders
    "load_gamma_from_checkpoint",
    "load_prior_from_checkpoint",
    "load_torchdecoder",
    # plotting
    "plot_avg_iters_vs_per",
    "plot_fr_vs_per",
    "plot_iters_distribution",
    "plot_ler_vs_per",
    "plot_smr_vs_per",
]
