import sys
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, DictConfig
from qecdec import RotatedSurfaceCode_Memory

_PYTORCH_ROOT = Path(__file__).resolve().parent.parent
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
    return Path(__file__).resolve().parent.parent / "datasets" / "rotated_surface_code_memory_Z" / f"d={d}_rounds={rounds}"


def main():
    cfg = load_config()
    print(">>>>>> Config:")
    print(OmegaConf.to_yaml(cfg))

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
        get_data_dir(cfg.qec),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=cfg.early_stopping.min_delta,
        patience=cfg.early_stopping.patience,
        verbose=True,
        mode="min",
    )
    epoch_summary = EpochSummary()
    tb_logger = TensorBoardLogger(
        save_dir="",
        log_graph=cfg.tb_logger.log_graph,
    )
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.max_epochs,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        callbacks=[early_stopping, epoch_summary],
        logger=tb_logger,
    )
    trainer.fit(decoder, datamodule=datamodule)


if __name__ == "__main__":
    sys.path.append(str(_PYTORCH_ROOT))
    from src.dataset import DecodingDataModule
    from src.lightning_module import DecodingModule
    from src.callbacks import EpochSummary

    main()
