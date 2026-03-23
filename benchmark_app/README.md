# Benchmark app

Streamlit UI for Monte Carlo benchmarking of trained PyTorch QEC decoders.

## Running

From the repo root:

```bash
uv run --project benchmark_app streamlit run benchmark_app/app.py
```

Or with pip (after `pip install -e benchmark_app`):

```bash
streamlit run benchmark_app/app.py
```

## Requirements

- Trained PyTorch decoder checkpoints in the configured runs directory (see `constants.py`)
