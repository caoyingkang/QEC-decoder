# Benchmark app

Streamlit UI for Monte Carlo benchmarking of QEC decoders.

## Running

From the repo root:

```bash
uv run streamlit run benchmark-app/app.py
```

## Layout


| Path                           | Contents                                                       |
| ------------------------------ | -------------------------------------------------------------- |
| `app.py`                   | Streamlit app entry point.                                     |
| `ui.py`                    | Sidebar / task-selection UI components.                        |
| `learned_decoders.py`      | Registers learned decoders (e.g. `LearnedDMemBP`) into qecdec. |
| `default_task_params.json` | Default decoder hyperparameters.                               |
| `results/`                 | CSV benchmark results.                                         |
| `notebooks/`               | Plotting notebooks.                                            |
