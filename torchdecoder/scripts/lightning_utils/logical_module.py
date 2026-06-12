"""Lightning module for logical decoder models (predict observables directly)."""

import math

import lightning as L
from omegaconf import DictConfig
import torch

from qecdec.circuits import QECCircuit
from torchdecoder_core.losses import LossResult, build_loss_fn
from torchdecoder_core.metrics import LogicalDecodingMetric
from torchdecoder_core.models import build_logical_decoder_model


class LogicalDecodingModule(L.LightningModule):
    def __init__(
        self,
        circuit: QECCircuit,
        *,
        model_cfg: DictConfig,
        loss_cfg: DictConfig,
        optim_cfg: DictConfig,
        compile_mode: str | None,
    ):
        """
        Parameters
        ----------
            circuit : QECCircuit
                The QEC circuit to decode (provides geometry, chkmat, obsmat).

            model_cfg : DictConfig
                Configuration for the logical decoder model.

            loss_cfg : DictConfig
                Configuration for the loss function.

            optim_cfg : DictConfig
                Configuration for the optimizer: `lr`, `weight_decay`,
                `warmup_steps`, `final_lr_ratio` (AdamW + linear warmup +
                cosine decay to `final_lr_ratio` of the peak lr).

            compile_mode : str | None
                Mode in torch.compile to optimize the decoder model.
                If None, no compilation is performed.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["circuit"])

        self.model = build_logical_decoder_model(circuit, model_cfg)
        if compile_mode is not None:
            self.model.compile(mode=compile_mode, fullgraph=True)

        self.loss_fn = build_loss_fn(circuit.chkmat, circuit.obsmat, loss_cfg)
        self.metric = LogicalDecodingMetric()

        # No example_input_array: ModelSummary's FLOP counter chokes on the
        # BatchNorm decomposition during its example forward pass (and we don't
        # log the model graph to tensorboard).

    def forward(self, syndromes: torch.Tensor):
        return self.model(syndromes)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = (
            batch  # (batch_size, num_chks), (batch_size, num_obsers), int
        )
        logits: torch.Tensor = self(syndromes)  # (batch_size, num_obsers), float
        result: LossResult = self.loss_fn(logits, observables)
        self.log("train_loss", result.loss, on_step=True, on_epoch=True, prog_bar=True)
        return result.loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = (
            batch  # (batch_size, num_chks), (batch_size, num_obsers), int
        )
        logits: torch.Tensor = self(syndromes)  # (batch_size, num_obsers), float
        result: LossResult = self.loss_fn(logits, observables)
        self.log("val_loss", result.loss, on_step=False, on_epoch=True)
        self.metric.update(logits, observables)
        return result.loss

    def on_validation_epoch_end(self):
        val_metrics = self.metric.compute()
        self.log_dict(val_metrics)
        self.metric.reset()

        if self.trainer.sanity_checking:
            self.print("\n--- Pre-Train Validation Summary ---")
        else:
            self.print(
                f"\n--- Epoch {self.trainer.current_epoch} Validation Summary ---"
            )
        self.print(f"Val Loss: {self.trainer.callback_metrics['val_loss']:.5f}")
        self.print(
            f"  Logical Success Rate: {val_metrics['logical_success_rate'] * 100:.2f}%"
        )
        self.print()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        syndromes, observables = (
            batch  # (batch_size, num_chks), (batch_size, num_obsers), int
        )
        logits = self(syndromes)  # (batch_size, num_obsers), float
        self.metric.update(logits, observables)

    def on_test_epoch_end(self):
        test_metrics = self.metric.compute()
        self.log_dict(test_metrics)
        self.metric.reset()

        self.print("\n--- Test Summary ---")
        self.print(
            f"  Logical Success Rate: {test_metrics['logical_success_rate'] * 100:.2f}%"
        )
        self.print()

    def configure_optimizers(self):
        optcfg = self.hparams.optim_cfg
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=optcfg.lr, weight_decay=optcfg.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        if not math.isfinite(total_steps):
            raise ValueError(
                "Cannot size the cosine lr schedule: the streaming train dataloader "
                "has no length, so trainer.max_steps must be set."
            )
        warmup_steps = optcfg.warmup_steps
        final_ratio = optcfg.final_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
            return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
