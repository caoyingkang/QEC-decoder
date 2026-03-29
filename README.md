# QEC-decoder

A monorepository that provides:

- a unified library for QEC decoding (see [packages/qecdec/README.md](packages/qecdec/README.md));
- a PyTorch stack for training decoders with learnable parameters (see [packages/torchdecoder-core/README.md](packages/torchdecoder-core/README.md) for core components and [torchdecoder/README.md](torchdecoder/README.md) for training entrypoints).
- a Streamlit UI for Monte Carlo benchmarking of PyTorch decoders (see [benchmark-app/README.md](benchmark-app/README.md)).

## Monorepo layout

| Path | Role |
|------|------|
| `packages/qecdec` | Core library: decoders (Rust implementation + Python bindings), QEC experiments, sliding window decoding, sinter integration. |
| `packages/torchdecoder-core` | PyTorch decoder models, loss functions, metrics, and dataset helpers |
| `torchdecoder` | PyTorch training/testing entrypoints and configs |
| `benchmark-app` | Streamlit UI for Monte Carlo benchmarking of PyTorch decoders |
| `misc` | Miscellaneous assets and ad hoc tooling not wired into the main packages |

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (for Python package and project management)
- [Rust](https://www.rust-lang.org/tools/install) (to build the `qecdec` package)

## Setup with uv (manual)

After cloning the repository, from the repository root:

```bash
uv sync
```

This will create a virtual environment in `.venv`, resolve the workspace, and build/install member packages `qecdec` and `torchdecoder_core`.

(Optional) By default, `uv sync` triggers a debug build of the Rust extension of `qecdec`. For a release build, run:

```bash
cd packages/qecdec
uvx maturin develop --release
```

(Optional) To run the notebooks in `packages/qecdec/notebooks/`, include extra dependencies by running from the repository root:

```bash
uv sync --all-packages --group qecdec-notebooks
```

## Dev container (alternative to manual setup)

This repository includes a VS Code Dev Container under `.devcontainer/`. It requires GPU access on the host machine.
