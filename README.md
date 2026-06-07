# QEC-decoder

A monorepository that provides:

- a unified library for QEC decoding (see [packages/qecdec/README.md](packages/qecdec/README.md));
- a Monte Carlo benchmarking suite for QEC decoders (see [packages/qecbench/README.md](packages/qecbench/README.md));
- a PyTorch stack for training decoders with learnable parameters (see [packages/torchdecoder-core/README.md](packages/torchdecoder-core/README.md) for core components and [torchdecoder/README.md](torchdecoder/README.md) for training entrypoints).
- a Streamlit UI interface for the Monte Carlo benchmarking suite (see [benchmark-app/README.md](benchmark-app/README.md)).

## Monorepo layout


| Path                         | Role                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `packages/qecdec`            | Core library: decoders (Rust implementation + Python bindings), QEC circuits (via stim), sinter integration. |
| `packages/qecbench`          | Monte Carlo benchmarking suite for QEC decoders (headless API).                                              |
| `packages/torchdecoder-core` | PyTorch decoder models, loss functions, metrics, dataset helpers, and qecdec adapters.                       |
| `torchdecoder`               | PyTorch training/testing entrypoints and configs.                                                            |
| `benchmark-app`              | Streamlit UI for Monte Carlo benchmarking of decoders.                                                       |
| `circuits`                   | Pre-generated `.stim` files for certain QEC circuits.                                                        |
| `misc`                       | Miscellaneous assets and ad hoc tooling not wired into the main packages.                                    |


## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (for Python package and project management)
- [Rust](https://www.rust-lang.org/tools/install) (to build the `qecdec` package)

## Setup with uv (manual)

After cloning the repository, from the repository root:

```bash
uv sync
```

This will create a virtual environment in `.venv`, resolve the workspace, and build/install all member packages (`qecdec`, `qecbench`, `torchdecoder_core`).

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

## Running unit tests

```bash
uv run pytest packages/*/tests -v
```
