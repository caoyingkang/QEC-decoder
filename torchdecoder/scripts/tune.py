"""
Optuna hyperparameter tuning script for PyTorch decoders.

Usage: (from torchdecoder/scripts directory)
    uv run python tune.py --config <path/to/config.yaml> [options]

Options:
    --config       : Path to base config YAML file (required)
    --n-trials     : Number of Optuna trials (default: 50)
    --max-epochs   : Override max_epochs per trial for faster search (default: use config value)
    --study-name   : Optuna study name (default: "hpo")
    --storage      : Optuna storage URL, e.g. sqlite:///hpo.db (default: in-memory)

Examples:
    uv run python tune.py --config configs/train_MultiDMemBP_ConvergenceAwareLoss_d=7.yaml
    uv run python tune.py --config configs/train_MultiDMemBP_ConvergenceAwareLoss_d=7.yaml --n-trials 100 --max-epochs 20
    uv run python tune.py --config configs/train_MultiDMemBP_ConvergenceAwareLoss_d=7.yaml --storage sqlite:///hpo.db
"""

import argparse
import os
from pathlib import Path
import warnings

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from omegaconf import OmegaConf, DictConfig
from qecdec.experiments import RotatedSurfaceCode_Memory

from lightning_utils import DecodingDataModule, DecodingModule

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def load_base_config(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    if cfg.data.num_workers is None:
        cfg.data.num_workers = max(1, (os.cpu_count() or 1) - 1)
    return cfg


def get_data_dir(qec_cfg: DictConfig) -> Path:
    return DATASETS_ROOT.joinpath(
        f"{qec_cfg.code}_{qec_cfg.noise_model}",
        f"d={qec_cfg.d}_rounds={qec_cfg.rounds}_basis={qec_cfg.basis}",
    )


def create_experiment(qec_cfg: DictConfig):
    if (
        qec_cfg.code == "RotatedSurfaceCode"
        and qec_cfg.noise_model == "Phenomenological"
    ):
        return RotatedSurfaceCode_Memory(
            d=qec_cfg.d,
            rounds=qec_cfg.rounds,
            basis=qec_cfg.basis,
            data_qubit_error_rate=qec_cfg.p,
            meas_error_rate=qec_cfg.p,
        )
    raise ValueError(
        f"Unsupported combination: {qec_cfg.code} + {qec_cfg.noise_model}"
    )


def suggest_hyperparameters(trial: optuna.Trial, cfg: DictConfig) -> DictConfig:
    """Create a mutable config copy with Optuna-suggested hyperparameters."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    cfg.data.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    cfg.optim.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg.loss.beta = trial.suggest_float("beta", 0.0, 1.0)
    cfg.loss.focal_gamma = trial.suggest_float("focal_gamma", 0.0, 5.0)
    cfg.optim.lr_scheduler.factor = trial.suggest_float("lr_scheduler_factor", 0.1, 0.5)
    cfg.optim.lr_scheduler.patience = trial.suggest_int("lr_scheduler_patience", 2, 6)

    return cfg


def objective(
    trial: optuna.Trial,
    base_cfg: DictConfig,
    expmt,
    data_dir: Path,
    max_epochs: int | None,
) -> float:
    cfg = suggest_hyperparameters(trial, base_cfg)

    epochs = max_epochs if max_epochs is not None else cfg.trainer.max_epochs

    decoder = DecodingModule(
        expmt.chkmat,
        expmt.obsmat,
        expmt.prior,
        model_cfg=cfg.model,
        loss_cfg=cfg.loss,
        optim_cfg=cfg.optim,
        compile_mode=cfg.compile_mode,
    )
    datamodule = DecodingDataModule(
        data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    pruning_callback = PyTorchLightningPruningCallback(
        trial, monitor="strict_success_rate"
    )
    early_stopping_callback = EarlyStopping(
        monitor="strict_success_rate",
        min_delta=cfg.early_stopping.min_delta,
        patience=cfg.early_stopping.patience,
        mode="max",
    )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        max_epochs=epochs,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[early_stopping_callback, pruning_callback],
        logger=False,
    )

    trainer.fit(decoder, datamodule=datamodule)

    return trainer.callback_metrics.get("strict_success_rate", 0.0).item()


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for PyTorch decoders"
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to base config YAML file"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs per trial for faster search",
    )
    parser.add_argument(
        "--study-name", type=str, default="hpo", help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///hpo.db)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_cfg = load_base_config(config_path)
    print(">>>>>> Base config:")
    print(OmegaConf.to_yaml(base_cfg))

    data_dir = get_data_dir(base_cfg.qec)
    print(f">>>>>> Data directory: {data_dir}")

    expmt = create_experiment(base_cfg.qec)
    print(f">>>>>> Number of error mechanisms: {expmt.num_error_mechanisms}")
    print(f">>>>>> Number of detectors: {expmt.num_detectors}")
    print(f">>>>>> Number of observables: {expmt.num_observables}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, base_cfg, expmt, data_dir, args.max_epochs),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print("Best trial:")
    print(f"  Value (strict_success_rate): {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="`isinstance\\(treespec, LeafSpec\\)` is deprecated",
    )

    main()
