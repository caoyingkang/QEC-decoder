# Benchmark app

Streamlit UI for Monte Carlo benchmarking of trained PyTorch QEC decoders.

## Running

From the repo root:

```bash
uv run streamlit run benchmark-app/app.py
```

## Headless usage (for long-running jobs / remote servers)

The custom Monte Carlo benchmark can also be driven from a Python script with
no Streamlit and no browser. This is the recommended path for jobs that take
hours and need to survive an SSH disconnect.

The entry point is `bench.custom_bench.run.run_custom_benchmark`. See
`scripts/run_example.py` for a complete example. The typical invocation is:

```bash
uv run python benchmark-app/scripts/run_example.py
```

The example script includes a small `sys.path` preamble that makes the
top-level `bench`, `constants`, `experiment_factory`, and `torchdecoder_utils`
modules importable regardless of the current working directory. Copy that
preamble when you write your own runner script.

To detach from the terminal:

```bash
# Inside tmux / screen
tmux new -s bench
uv run python benchmark-app/scripts/run_example.py
# Ctrl-b d to detach; tmux attach -t bench to reattach

# Or with nohup, redirecting progress output to a log file
nohup uv run python benchmark-app/scripts/run_example.py > run.log 2>&1 &
```

Output CSVs are written under `benchmark-app/baselines-results/...` (baselines)
or each PyTorch run directory (`custom_benchmark.csv`). Re-running with the
same parameters resumes from the existing CSVs, so Ctrl-C and rerun is safe.

The same `run_custom_benchmark` function is used internally by the Streamlit
page, so the headless and UI paths produce identical results.
