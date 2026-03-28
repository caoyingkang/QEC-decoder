# qecdec

Core **q**uantum **e**rror **c**orrection **dec**oding library: Rust implementations of decoders (BP, DMemBP, UnionFind, etc.) exposed to Python via PyO3/maturin, plus Python APIs for memory experiments, sliding-window decoding, and `sinter` integration.

This package is a member of the repository’s [uv](https://docs.astral.sh/uv/) workspace; install it from the repository root with `uv sync` (see the root [README.md](../../README.md)).

## Package layout

| Path | Contents |
|------|----------|
| **`Cargo.toml`**, **`pyproject.toml`** | Rust and Python/maturin project metadata. |
| **`src/`** | Rust crate: decoder implementations (BP, DMemBP, UnionFind, etc.). |
| **`python/qecdec/`** | Source code for the Python package `qecdec`. |
| **`python/qecdec/decoders/`** | Python-facing decoder APIs backed by the Rust module. |
| **`python/qecdec/experiments/`** | Circuit for memory experiments. |
| **`python/qecdec/sinter_utils/`** | Helpers to plug decoders into [sinter](https://pypi.org/project/sinter/). |
| **`python/qecdec/slwin/`** | Helpers to use decoders in sliding-window decoding. |
| **`notebooks/`** | Example Jupyter notebooks. |

Extra dependencies for `notebooks/` are listed under `[dependency-groups]` in `pyproject.toml`. To include them, run the following command from the repository root:
```bash
uv sync --all-packages --group qecdec-notebooks
```
