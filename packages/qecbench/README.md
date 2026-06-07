# qecbench

Headless API of a Monte Carlo benchmarking suite for QEC decoders.

## Running tests

The test suite under `tests/` uses [pytest](https://pytest.org). From the repository root, install the `test` dependency group and run the suite:

```bash
uv sync --group test
uv run pytest packages/qecbench/tests -v
```

## Usage example

```python
from qecbench import CollectorParams, TaskMetadata, run_benchmark

task = TaskMetadata(
    circuit_name="RotatedSurfaceCode_Phenom",
    circuit_params={"d": 5, "rounds": 5, "basis": "Z"},
    error_rate=0.01,
    decoder_name="MemBP",
    decoder_params={"max_iter": 50, "gamma": 0.2},
)
collector = CollectorParams(
    batch_size=128,
    shots_cap=10_000_000,
    errors_cap=100,
    num_parallel_workers=0,  # 0 = serial; >0 = multiprocessing
)

stats = run_benchmark(task, collector, csv_path="results.csv")
print(stats)
```
