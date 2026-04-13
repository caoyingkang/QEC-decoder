"""
Python script to train a PyTorch decoder.

Usage: (from torchdecoder/scripts directory)
    uv run python train.py --config <path/to/config.yaml> [options] [overrides...]

Options:
    -h, --help : Show help message and exit
    --profile : Profile a few training steps

Examples: (from the torchdecoder/scripts directory)
    uv run python train.py --config configs/train_LearnedDMemBP_UniformIterationLoss_d=5.yaml
    uv run python train.py --config configs/train_MultiDMemBP_UniformIterationLoss_d=5.yaml loss.beta=0.0 model.mlp.activation=ReLU
    uv run python train.py --config configs/train_MultiDMemBP_ConvergenceAwareLoss_d=5.yaml --profile
"""

import argparse
import os
from pathlib import Path
import warnings

from tabulate import tabulate
import humanize
import lightning as L
from lightning.pytorch.callbacks import (
    ModelSummary,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError
from lightning_utils import (
    CurriculumCallback,
    DecodingDataModule,
    DecodingModule,
)
from utils import create_experiment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
RUNS_ROOT = PROJECT_ROOT / "runs"


def load_config(path: Path, overrides: list[str]) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    base_cfg = OmegaConf.load(path)
    OmegaConf.resolve(base_cfg)
    OmegaConf.set_struct(base_cfg, True)  # forbid creation of new keys
    overrides_cfg = OmegaConf.from_cli(overrides)

    try:
        cfg = OmegaConf.merge(base_cfg, overrides_cfg)
    except ConfigKeyError as e:
        print(
            f"Invalid CLI override(s). Only keys that exist in the base config can be overridden.\n {e}"
        )
        exit(1)

    if cfg.data.num_workers is None:
        cfg.data.num_workers = max(1, (os.cpu_count() or 1) - 1)

    OmegaConf.set_readonly(cfg, True)  # forbid modification from now on
    return cfg


def get_data_dir(qec_cfg: DictConfig) -> Path:
    code = qec_cfg.code
    noise_model = qec_cfg.noise_model
    d = qec_cfg.d
    rounds = qec_cfg.rounds
    basis = qec_cfg.basis
    return DATASETS_ROOT.joinpath(
        f"{code}_{noise_model}",
        f"d={d}_rounds={rounds}_basis={basis}",
    )


def get_run_dir(cfg: DictConfig) -> Path:
    code = cfg.qec.code
    noise_model = cfg.qec.noise_model
    d = cfg.qec.d
    rounds = cfg.qec.rounds
    basis = cfg.qec.basis
    model_name = cfg.model.name
    base_dir = RUNS_ROOT.joinpath(
        f"{code}_{noise_model}",
        f"d={d}_rounds={rounds}_basis={basis}",
        model_name,
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    run_ids = set[int]()
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            run_ids.add(int(p.name[4:]))
    new_id = 0
    while new_id in run_ids:
        new_id += 1
    return base_dir / f"run_{new_id}"


def print_params(module: L.LightningModule):
    table_data = []
    total_numel = 0
    total_bytes = 0
    for name, param in module.named_parameters():
        numel = param.numel()
        bytes = numel * param.element_size()
        total_numel += numel
        total_bytes += bytes
        table_data.append(
            [
                name,
                str(param.dtype).split(".")[-1],
                list(param.size()),
                f"{numel:,}",
                humanize.naturalsize(bytes, binary=True),
            ]
        )
    table_data.append(
        [
            "Total",
            "",
            "",
            f"{total_numel:,}",
            humanize.naturalsize(total_bytes, binary=True),
        ]
    )
    print(
        tabulate(
            table_data,
            headers=["name", "dtype", "size", "numel", "bytes"],
            tablefmt="fancy_grid",
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch decoder")
    parser.add_argument(
        "--config", required=True, type=str, help="Path to config YAML file"
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args, overrides = parser.parse_known_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path, overrides)
    qec_cfg = cfg.qec
    print(">>>>>> Config:")
    print(OmegaConf.to_yaml(cfg))

    data_dir = get_data_dir(qec_cfg)
    run_dir = get_run_dir(cfg)
    print(f">>>>>> Data directory: {data_dir}")
    print(f">>>>>> Run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    load_circuit_from_file = qec_cfg.code in ["BB_144_12_12"]
    expmt = create_experiment(
        qec_cfg.code,
        qec_cfg.noise_model,
        qec_cfg.d,
        qec_cfg.rounds,
        qec_cfg.basis,
        qec_cfg.p,
        load_circuit_from_file,
    )
    print(f">>>>>> Number of error mechanisms: {expmt.num_error_mechanisms}")
    print(f">>>>>> Number of detectors: {expmt.num_detectors}")
    print(f">>>>>> Number of observables: {expmt.num_observables}")
    decoder = DecodingModule(
        expmt.chkmat,
        expmt.obsmat,
        expmt.prior,
        model_cfg=cfg.model,
        loss_cfg=cfg.loss,
        optim_cfg=cfg.optim,
        compile_mode=cfg.compile_mode,
    )
    print(">>>>>> Parameters:")
    print_params(decoder)
    datamodule = DecodingDataModule(
        data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    early_stopping_callback = EarlyStopping(
        monitor="strict_success_rate",
        min_delta=cfg.early_stopping.min_delta,
        patience=cfg.early_stopping.patience,
        mode="max",
    )
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best_model",
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
    )
    tb_logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tb_logs",
        version="",
        log_graph=cfg.tb_logger.log_graph,
    )
    profiler = (
        PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(run_dir / "tb_logs" / "profile")
            ),
            profile_memory=True,
            track_memory=True,
            with_stack=True,
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=10, wait=5, warmup=5, active=10, repeat=1
            ),
        )
        if args.profile
        else None
    )
    callbacks = [
        ModelSummary(max_depth=-1),
        CurriculumCallback(),
        LearningRateMonitor(logging_interval="epoch"),
        early_stopping_callback,
        model_checkpoint_callback,
    ]
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs if profiler is None else 1,
        limit_train_batches=None if profiler is None else 50,
        limit_val_batches=None if profiler is None else 50,
        num_sanity_val_steps=-1 if profiler is None else 0,  # Pre-train validation
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        callbacks=callbacks,
        logger=tb_logger,
        enable_model_summary=False,  # We've already added model summary as a callback
        profiler=profiler,
    )

    # Start training
    trainer.fit(decoder, datamodule=datamodule)

    if early_stopping_callback.stopping_reason_message:
        print(
            f"Early stopping reason: {early_stopping_callback.stopping_reason_message}"
        )


if __name__ == "__main__":
    # Filter out some warnings generated by lightning package.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="`isinstance\\(treespec, LeafSpec\\)` is deprecated",
    )

    # torch.set_float32_matmul_precision("high")

    main()
