# torchdecoder

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) modules for neural-network-based QEC decoders, along with dataset builders, training/testing scripts, and configuration files.

This directory is a member of the repository’s [uv](https://docs.astral.sh/uv/) workspace, and is built on top of the packages [qecdec](../packages/qecdec) and [torchdecoder-core](../packages/torchdecoder-core). Install all dependencies from the repository root with `uv sync` (see the root [README.md](../README.md)).

## Building datasets

Training and testing expect PyTorch datasets under `datasets/`. This directory contains subdirectories for each QEC setting, each with a pre-filled `config.yaml` file that controls how the datasets are built.

To generate the datasets, from `torchdecoder/scripts/`:

```bash
uv run python build_datasets.py [options]
```

**Options**

- `-h`, `--help`: Show help message and exit.
- `--force`: Rebuild train, validation, and test datasets even if they already exist.


## Training

From `torchdecoder/scripts/`:

**Usage**

```bash
uv run python train.py --config <path/to/config.yaml> [options] [overrides...]
```

**Options**

- `-h`, `--help`: Show help message and exit.
- `--profile`: Profile a few training steps.

**Examples**

```bash
uv run python train.py --config configs/train_LearnedDMemBP_UniformIterationLoss_d=5.yaml
```
```bash
uv run python train.py --config configs/train_MultiDMemBP_UniformIterationLoss_d=5.yaml loss.beta=0.0 model.mlp.activation=ReLU
```
```bash
uv run python train.py --config configs/train_MultiDMemBP_ConvergenceAwareLoss_d=5.yaml --profile
```

## Testing

From `torchdecoder/scripts/`:

**Usage**

```bash
uv run python test.py --run-dir <path/to/run_directory>
```

`run_directory` is the folder created during training (under `runs/...`). It must contain a checkpoint file `checkpoints/best_model.ckpt`.

**Example**

```bash
uv run python test.py --run-dir ../runs/RotatedSurfaceCode_Phenomenological/d=5_rounds=5_basis=Z/MultiDMemBP/run_0
```

## Directory layout

| Path | Contents |
|------|----------|
| **`pyproject.toml`** | Python project metadata. |
| **`scripts/`** | Dataset builder, training and testing scripts. |
| **`scripts/configs/`** | Example training configuration files. |
| **`scripts/lightning_utils/`** | LightningModule and LightningDataModule. |
| **`datasets/`** | Dataset folders with preset configurations. |
| **`notebooks/`** | Ad hoc notebooks. |
| **`runs/`** | Directories for storing checkpoints and tensorboard logs. Will be created by training scripts. |
