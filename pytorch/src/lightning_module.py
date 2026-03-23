from typing import Optional

import numpy as np
import torch
import lightning as L
from omegaconf import DictConfig

from .models import build_decoder_model
from .losses import build_decoding_loss, LossResult
from .metric import IterativeDecodingMetric


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
        compile_mode: Optional[str],
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

            compile_mode : Optional[str]
                Mode in torch.compile to optimize the decoder model and the loss function.
                If None, no compilation is performed.
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
        self.model = build_decoder_model(chkmat, prior, model_cfg)
        if compile_mode is not None:
            self.model.compile(mode=compile_mode, fullgraph=True)

        # Set up loss function.
        self.loss_fn = build_decoding_loss(chkmat, obsmat, loss_cfg)
        if compile_mode is not None:
            self.loss_fn.compile(mode=compile_mode, fullgraph=True)

        # Set up metric.
        self.metric = IterativeDecodingMetric(chkmat, obsmat)

        # Set example input array for tensorboard to log model graph.
        self.example_input_array = torch.randint(0, 2, (1, chkmat.shape[0]))

    def forward(self, syndromes: torch.Tensor):
        return self.model(syndromes)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = batch  # (batch_size, num_chks), (batch_size, num_obsers), int
        llrs: torch.Tensor = self(syndromes)  # (num_iters, batch_size, num_vars), float
        result: LossResult = self.loss_fn(llrs, syndromes, observables)
        self.log('train_loss', result.loss, on_step=False, on_epoch=True)
        self.log('train_synd_loss', result.synd_loss, on_step=False, on_epoch=True)
        self.log('train_obser_loss', result.obser_loss, on_step=False, on_epoch=True)
        return result.loss

    def on_before_optimizer_step(self, optimizer):
        global_step = self.trainer.global_step
        if global_step % 100 == 1:  # Log every 100 steps to avoid slowing down training
            for name, p in self.model.named_parameters():
                # Inspect parameter values distribution
                self.logger.experiment.add_histogram(
                    tag=f"params/{name}",
                    values=p.detach(),
                    global_step=global_step
                )
                if p.grad is not None:
                    # Inspect gradient norm
                    self.log(f"grad_norm/{name}", torch.linalg.norm(p.grad.detach(), 2))
                    # Inspect gradient distribution
                    self.logger.experiment.add_histogram(
                        tag=f"grads/{name}",
                        values=p.grad.detach(),
                        global_step=global_step
                    )

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = batch  # (batch_size, num_chks), (batch_size, num_obsers), int
        llrs: torch.Tensor = self(syndromes)  # (num_iters, batch_size, num_vars), float
        result: LossResult = self.loss_fn(llrs, syndromes, observables)
        self.log('val_loss', result.loss, on_step=False, on_epoch=True)
        self.log('val_synd_loss', result.synd_loss, on_step=False, on_epoch=True)
        self.log('val_obser_loss', result.obser_loss, on_step=False, on_epoch=True)
        self.metric.update(llrs, syndromes, observables)
        return result.loss

    def on_validation_epoch_end(self):
        val_metrics = self.metric.compute()
        self.log_dict(val_metrics)
        self.metric.reset()

        if self.trainer.sanity_checking:
            self.print("\n--- Pre-Train Validation Summary ---")
        else:
            self.print(f"\n--- Epoch {self.trainer.current_epoch} Validation Summary ---")
        self.print(f"Val Loss: {self.trainer.callback_metrics['val_loss']:.5f}")
        self.print("Val Metrics:")
        self.print(f"  Convergence Rate: {val_metrics['convergence_rate'] * 100:.2f}%")
        self.print(f"  Logical Success Rate: {val_metrics['logical_success_rate'] * 100:.2f}%")
        self.print(f"  Strict Success Rate: {val_metrics['strict_success_rate'] * 100:.2f}%")
        self.print(f"  Accidental Success Rate: {val_metrics['accidental_success_rate'] * 100:.2f}%")
        self.print(f"  Success Rate on Convergence: {val_metrics['success_rate_on_convergence'] * 100:.2f}%")
        self.print(f"  Average Iterations: {val_metrics['avg_iters']:.2f}")
        self.print(f"  Average Iterations on Convergence: {val_metrics['avg_iters_on_convergence']:.2f}")
        self.print()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = batch  # (batch_size, num_chks), (batch_size, num_obsers), int
        llrs = self(syndromes)  # (num_iters, batch_size, num_vars), float
        self.metric.update(llrs, syndromes, observables)

    def on_test_epoch_end(self):
        test_metrics = self.metric.compute()
        self.log_dict(test_metrics)
        self.metric.reset()

        self.print("\n--- Test Summary ---")
        self.print(f"  Convergence Rate: {test_metrics['convergence_rate'] * 100:.2f}%")
        self.print(f"  Logical Success Rate: {test_metrics['logical_success_rate'] * 100:.2f}%")
        self.print(f"  Strict Success Rate: {test_metrics['strict_success_rate'] * 100:.2f}%")
        self.print(f"  Accidental Success Rate: {test_metrics['accidental_success_rate'] * 100:.2f}%")
        self.print(f"  Success Rate on Convergence: {test_metrics['success_rate_on_convergence'] * 100:.2f}%")
        self.print(f"  Average Iterations: {test_metrics['avg_iters']:.2f}")
        self.print(f"  Average Iterations on Convergence: {test_metrics['avg_iters_on_convergence']:.2f}")
        self.print()

    def configure_optimizers(self):
        optcfg = self.hparams.optim_cfg
        lrcfg = optcfg.lr_scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=optcfg.lr)
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
