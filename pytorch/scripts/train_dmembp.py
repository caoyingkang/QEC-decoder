import sys
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, DictConfig
from qecdec import RotatedSurfaceCode_Memory

PYTORCH_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PYTORCH_ROOT / "datasets"
RUNS_ROOT = PYTORCH_ROOT / "runs"
DEFAULT_CONFIG_PATH = PYTORCH_ROOT / "configs" / (Path(__file__).stem + ".yaml")


def load_config() -> DictConfig:
    cli_args = OmegaConf.from_cli()
    config_path = Path(cli_args.config) if "config" in cli_args else DEFAULT_CONFIG_PATH
    base_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, cli_args)
    if cfg.data.num_workers is None:
        cfg.data.num_workers = os.cpu_count()
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
    cfg = load_config()
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
        ],
        logger=tb_logger,
    )
    trainer.fit(decoder, datamodule=datamodule)


if __name__ == "__main__":
    sys.path.append(str(PYTORCH_ROOT))
    from src.dataset import DecodingDataModule
    from src.lightning_module import DecodingModule
    from src.callbacks import EpochSummary

    main()
