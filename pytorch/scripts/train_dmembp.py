import sys
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, DictConfig
from qecdec import RotatedSurfaceCode_Memory

_PYTORCH_ROOT = Path(__file__).resolve().parent.parent
_DATASETS_ROOT = _PYTORCH_ROOT / "datasets"
_RUNS_ROOT = _PYTORCH_ROOT / "runs"
_DEFAULT_CONFIG_PATH = _PYTORCH_ROOT / "configs" / (Path(__file__).stem + ".yaml")


def load_config() -> DictConfig:
    cli_args = OmegaConf.from_cli()
    config_path = Path(cli_args.config) if "config" in cli_args else _DEFAULT_CONFIG_PATH
    base_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, cli_args)
    if cfg.data.num_workers is None:
        cfg.data.num_workers = os.cpu_count()
    return cfg


def get_data_dir(qec_cfg: DictConfig) -> Path:
    d = qec_cfg.d
    rounds = qec_cfg.rounds
    basis = qec_cfg.basis
    return _DATASETS_ROOT / "rotated_surface_code_memory" / f"d={d}_rounds={rounds}_basis={basis}"


def get_run_dir(cfg: DictConfig) -> Path:
    d = cfg.qec.d
    rounds = cfg.qec.rounds
    basis = cfg.qec.basis
    return _RUNS_ROOT / "rotated_surface_code_memory" / f"d={d}_rounds={rounds}_basis={basis}" / cfg.model.name


def main():
    cfg = load_config()
    print(">>>>>> Config:")
    print(OmegaConf.to_yaml(cfg))

    data_dir = get_data_dir(cfg.qec)
    run_dir = get_run_dir(cfg)

    expmt = RotatedSurfaceCode_Memory(
        d=cfg.qec.d,
        rounds=cfg.qec.rounds,
        basis=cfg.qec.basis,
        data_qubit_error_rate=cfg.qec.data_qubit_error_rate,
        meas_error_rate=cfg.qec.meas_error_rate,
    )
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
        dirpath=run_dir,
        filename="best_model",
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
    )
    tb_logger = TensorBoardLogger(
        save_dir="",
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
        ],
        logger=tb_logger,
    )
    trainer.fit(decoder, datamodule=datamodule)


if __name__ == "__main__":
    sys.path.append(str(_PYTORCH_ROOT))
    from src.dataset import DecodingDataModule
    from src.lightning_module import DecodingModule
    from src.callbacks import EpochSummary

    main()
