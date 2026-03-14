"""
Python script to train a PyTorch decoder.

Usage:
    python train.py --config <path/to/config.yaml> [overrides...]

Examples: (Assuming run in the pytorch/scripts directory)
    python train.py --config configs/train_LearnedDMemBP.yaml
    python train.py --config configs/train_MultiDMemBP.yaml qec.d=11 qec.rounds=11 model.mlp.activation=ReLU
"""

import argparse
import os
import sys
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError
from qecdec import RotatedSurfaceCode_Memory

PYTORCH_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PYTORCH_ROOT / "datasets"
RUNS_ROOT = PYTORCH_ROOT / "runs"


def load_config(path: Path, overrides: list[str]) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    base_cfg = OmegaConf.load(path)
    OmegaConf.set_struct(base_cfg, True)  # forbid creation of new keys
    overrides_cfg = OmegaConf.from_cli(overrides)

    try:
        cfg = OmegaConf.merge(base_cfg, overrides_cfg)
    except ConfigKeyError as e:
        print(f"Invalid CLI override(s). Only keys that exist in the base config can be overridden.\n {e}")
        exit(1)

    if cfg.data.num_workers is None:
        cfg.data.num_workers = os.cpu_count()

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


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch decoder")
    parser.add_argument("--config", required=True, type=str, help="Path to config YAML file")
    args, overrides = parser.parse_known_args()

    cfg = load_config(Path(args.config), overrides)
    qec_cfg = cfg.qec
    print(">>>>>> Config:")
    print(OmegaConf.to_yaml(cfg))

    data_dir = get_data_dir(qec_cfg)
    run_dir = get_run_dir(cfg)
    print(f">>>>>> Data directory: {data_dir}")
    print(f">>>>>> Run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    if qec_cfg.code == "RotatedSurfaceCode" and qec_cfg.noise_model == "Phenomenological":
        expmt = RotatedSurfaceCode_Memory(
            d=qec_cfg.d,
            rounds=qec_cfg.rounds,
            basis=qec_cfg.basis,
            data_qubit_error_rate=qec_cfg.p,
            meas_error_rate=qec_cfg.p,
        )
    else:
        raise ValueError(f"Unsupported combination: {qec_cfg.code} + {qec_cfg.noise_model}")
    print(f">>>>>> Number of error mechanisms: {expmt.num_error_mechanisms}")
    print(f">>>>>> Number of detectors: {expmt.num_detectors}")
    print(f">>>>>> Number of observables: {expmt.num_observables}")
    decoder = DecodingModule(
        expmt.chkmat, expmt.obsmat, expmt.prior,
        model_cfg=cfg.model,
        loss_cfg=cfg.loss,
        optim_cfg=cfg.optim,
    )
    datamodule = DecodingDataModule(
        data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=cfg.early_stopping.min_delta,
        patience=cfg.early_stopping.patience,
        verbose=True,
        mode="min",
    )
    epoch_summary_callback = EpochSummary()
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best_model",
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tb_logs",
        version="",
        log_graph=cfg.tb_logger.log_graph,
    )
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        callbacks=[
            early_stopping_callback,
            epoch_summary_callback,
            model_checkpoint_callback,
            lr_monitor_callback,
        ],
        logger=tb_logger,
    )
    # Run full validation before training
    trainer.validate(decoder, datamodule=datamodule)

    # Start training
    trainer.fit(decoder, datamodule=datamodule)


if __name__ == "__main__":
    sys.path.append(str(PYTORCH_ROOT))
    from src.dataset import DecodingDataModule
    from src.lightning_module import DecodingModule
    from src.callbacks import EpochSummary

    main()
