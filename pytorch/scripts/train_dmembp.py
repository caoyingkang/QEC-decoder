from src.dataset import DecodingDataModule
from src.lightning_module import DecodingModule
import sys
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from qecdec import RotatedSurfaceCode_Memory

# Add pytorch directory to sys.path.
sys.path.append(str(Path(__file__).resolve().parent.parent))


# QEC experiment configuration.
qec_cfg = dict(
    d=5,
    rounds=5,
    basis='Z',
    data_qubit_error_rate=0.01,
    meas_error_rate=0.01,
)

# Model configuration.
model_cfg = dict(
    name="learned_dmembp",
    num_iters=5,
    min_impl_method="smooth",
    sign_impl_method="smooth",
)

# Loss function configuration.
loss_cfg = dict(
    beta=0.8,
    skip_iters=0,
)

# Optimizer configuration.
optim_cfg = dict(
    lr=0.002,
)

# Early stopping configuration.
early_stopping_cfg = dict(
    min_delta=1e-4,
    patience=5,
)

# Tensorboard logger configuration.
tb_logger_cfg = dict(
    save_dir="",
    log_graph=False,
)

# Data module configuration.
data_cfg = dict(
    batch_size=256,
    num_workers=os.cpu_count(),
)

# Trainer configuration.
train_cfg = dict(
    accelerator="cpu",  # "gpu", "auto"
    max_epochs=20,
    enable_progress_bar=True,
)


def get_data_dir(qec_cfg: dict) -> Path:
    d = qec_cfg['d']
    rounds = qec_cfg['rounds']
    p = qec_cfg['data_qubit_error_rate']
    return Path(__file__).resolve().parent.parent / "datasets" / "rotated_surface_code_memory_Z" / f"d={d}_rounds={rounds}_p={p}"


def main():
    expmt = RotatedSurfaceCode_Memory(**qec_cfg)
    print(f">>>>>> Number of error mechanisms: {expmt.num_error_mechanisms}")
    print(f">>>>>> Number of detectors: {expmt.num_detectors}")
    print(f">>>>>> Number of observables: {expmt.num_observables}")

    decoder = DecodingModule(
        expmt.chkmat, expmt.obsmat, expmt.prior,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        optim_cfg=optim_cfg,
    )

    datamodule = DecodingDataModule(get_data_dir(qec_cfg), **data_cfg)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True, **early_stopping_cfg)
    tb_logger = TensorBoardLogger(**tb_logger_cfg)
    trainer = L.Trainer(
        callbacks=[early_stopping],
        logger=tb_logger,
        **train_cfg,
    )

    trainer.fit(decoder, datamodule=datamodule)


if __name__ == "__main__":
    main()
