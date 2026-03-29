# torchdecoder-core

Shared PyTorch building blocks for neural-network-based QEC decoders: model definitions, loss functions, metrics, dataset classes, and utility functions.

This package is a member of the repository’s [uv](https://docs.astral.sh/uv/) workspace; install it from the repository root with `uv sync` (see the root [README.md](../../README.md)).

This package does not contain dataset builders or training/testing scripts; see [torchdecoder](../../torchdecoder/) for that. This package is also consumed by [benchmark-app](../../benchmark-app/) for Monte Carlo benchmarking.


## Package layout

| Path | Contents |
|------|----------|
| **`pyproject.toml`** | Python project metadata. |
| **`src/torchdecoder_core/`** | Source code for the Python package `torchdecoder_core`. |
| **`src/torchdecoder_core/models/`** | Decoder model implementations and factory: `LearnedDMemBP`, `MultiDMemBP`. |
| **`src/torchdecoder_core/losses/`** | Loss functions and factory: `UniformIterationLoss`, `ConvergenceAwareLoss`. |
| **`src/torchdecoder_core/metrics/`** | Validation metrics. |
| **`src/torchdecoder_core/dataset/`** | Dataset classes. |
| **`src/torchdecoder_core/utils/`** | Helper modules and utility functions. |

## Definitions of decoding metrics

- **Convergence rate** — fraction of attempts where the decoder converged (i.e., the estimated error satisfied the syndrome).
- **Logical success rate** — fraction of attempts where logical observables were predicted correctly.
- **Strict success rate** — both converged and logically correct.
- **Accidental success rate** — not converged but logically correct anyway.
- **Success rate on convergence** — logically correct conditioned on convergence.
- **Average iterations** — mean iteration count.
- **Average iterations on convergence** — mean iteration count conditioned on convergence.
