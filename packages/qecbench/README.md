# qecbench

Monte Carlo benchmark tool for QEC decoders — a customized lightweight replacement for [sinter](https://pypi.org/project/sinter/) with extra features.

## Features

- No sinter dependency.
- Finer-grained analysis than sinter, with iterative decoding in mind: benchmark not just decoder's performance in predicting logical observables, but also the histogram of iterations the decoder ran for each shot before returning.
- Multiprocessing can be turned on for faster Monte Carlo loop on CPU, and can be turned off when benchmarking PyTorch decoders on GPU to prevent resource issues on some platform.
- Manual control of the batch size for running the benchmark loop.

## Installation

As part of the monorepo (from repo root):

```bash
uv sync
```

Or install standalone:

```bash
pip install -e packages/qecbench
```

## Usage

```python
from qecbench import BenchmarkDecoder, DecodeResult, TaskMetadata, collect_stats
```

The user should implement or wrap the decoder as a subclass of `BenchmarkDecoder` (with a `decode(syndromes) -> DecodeResult` method), then call `collect_stats` with a stim detector error model, your decoder, and metadata for the task.
