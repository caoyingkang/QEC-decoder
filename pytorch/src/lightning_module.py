import numpy as np
import torch
import lightning as L
from omegaconf import DictConfig

from .models import build_decoder_model
from .loss import IterativeDecodingLoss
from .metric import DecodingMetric

EPS = 1e-6
BIG = 1e8
FLOAT_DTYPE = torch.float32


class DecodingModule(L.LightningModule):
    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
        prior: np.ndarray,
        *,
        model_cfg: DictConfig,
        loss_cfg: DictConfig,
        optim_cfg: DictConfig,
    ):
        """
        Parameters
        ----------
            chkmat : ndarray
                Parity-check matrix, shape=(num_chks, num_vars), integer ∈ {0,1} or bool

            obsmat : ndarray
                Observable matrix, shape=(num_obsers, num_vars), integer ∈ {0,1} or bool

            prior : ndarray
                Prior probabilities of errors, shape=(num_vars,), float

            model_cfg : DictConfig
                Configuration for the decoder model.

            loss_cfg : DictConfig
                Configuration for the loss function.

            optim_cfg : DictConfig
                Configuration for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['chkmat', 'obsmat', 'prior'])

        # Validate chkmat, obsmat, and prior.
        assert isinstance(chkmat, np.ndarray) and isinstance(obsmat, np.ndarray) and isinstance(prior, np.ndarray)
        assert np.issubdtype(chkmat.dtype, np.integer) or np.issubdtype(chkmat.dtype, np.bool_)
        assert np.issubdtype(obsmat.dtype, np.integer) or np.issubdtype(obsmat.dtype, np.bool_)
        assert np.issubdtype(prior.dtype, np.floating)
        assert chkmat.ndim == 2 and obsmat.ndim == 2 and prior.ndim == 1
        assert chkmat.shape[1] == obsmat.shape[1] == prior.shape[0]

        # Build decoder model.
        self.decoder = build_decoder_model(chkmat, prior, model_cfg)

        # Set up loss function and metric.
        self.loss_fn = IterativeDecodingLoss(
            chkmat, obsmat,
            beta=loss_cfg.beta,
            skip_iters=loss_cfg.skip_iters,
        )
        self.metric = DecodingMetric(chkmat, obsmat)

        # Set example input array for tensorboard to log model graph.
        self.example_input_array = torch.randint(0, 2, (1, chkmat.shape[0]))

    def forward(self, syndromes: torch.Tensor):
        return self.decoder(syndromes)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = batch  # (batch_size, num_chks), (batch_size, num_obsers)
        llrs = self(syndromes)  # (num_iters, batch_size, num_vars)
        loss = self.loss_fn(llrs, syndromes, observables)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = batch  # (batch_size, num_chks), (batch_size, num_obsers)
        llrs = self(syndromes)  # (num_iters, batch_size, num_vars)
        loss = self.loss_fn(llrs, syndromes, observables)
        self.log('val_loss', loss, prog_bar=True)
        self.metric.update(llrs.cpu(), syndromes.cpu(), observables.cpu())
        return loss

    def on_validation_epoch_end(self):
        val_metrics = self.metric.compute()
        self.log_dict(val_metrics)
        self.metric.reset()

    def configure_optimizers(self):
        optcfg = self.hparams.optim_cfg
        lrcfg = optcfg.lr_scheduler
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=optcfg.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lrcfg.factor,
            patience=lrcfg.patience,
            threshold=lrcfg.threshold,
            threshold_mode=lrcfg.threshold_mode,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }
