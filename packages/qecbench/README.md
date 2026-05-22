# qecbench

Monte Carlo benchmarking for QEC decoders, factored out of the repo's
`benchmark-app/` Streamlit UI so it can be driven from notebooks, headless
scripts, and CI as well.

`qecbench` depends only on [`qecdec`](../qecdec) and
[`torchdecoder-core`](../torchdecoder-core); it has no Streamlit imports and
no hardcoded data-file locations. Callers pass in pre-built
`qecdec.experiments.Experiment` objects and explicit checkpoint / CSV paths.

## Running tests

The test suite under `tests/` uses [pytest](https://pytest.org). From the repository root, install the `test` dependency group and run the suite:

```bash
uv sync --group test
uv run pytest packages/qecbench/tests -v
```

## Public API

```python
from qecbench import (
    # parameters
    BenchTaskParams, CollectorParams, TorchDecoderTask,
    # stats
    TaskMetadata, TaskStats,
    # decoders
    BenchmarkDecoder, DecodeResult,
    # collection + orchestration
    collect_stats,
    run_custom_benchmark, run_baseline_benchmark, run_torchdecoder_benchmark,
    # IO
    StatsSource, load_and_filter_stats,
    get_baseline_csv_path, get_torchdecoder_csv_path,
    # torch helpers
    load_torchdecoder, load_gamma_from_checkpoint, load_prior_from_checkpoint,
    # plotting
    plot_fr_vs_per, plot_ler_vs_per, plot_smr_vs_per,
    plot_avg_iters_vs_per, plot_iters_distribution,
    # decoder name lists
    BASELINE_DECODERS_GRAPHLIKE, BASELINE_DECODERS_HYPERGRAPH,
    DEFAULT_BASELINE_DECODERS_GRAPHLIKE, DEFAULT_BASELINE_DECODERS_HYPERGRAPH,
    ALL_BASELINE_DECODERS,
)
```
