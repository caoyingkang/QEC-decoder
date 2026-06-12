"""Custom Lightning callbacks for training QEC decoders."""

import lightning as L
import torch


class CurriculumCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.enabled = False

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if (
            hasattr(pl_module.loss_fn, "curriculum")
            and pl_module.loss_fn.curriculum is not None
        ):
            self.enabled = True
            pl_module.print(">>>>>> Sample-level curriculum learning enabled.")

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.enabled:
            pl_module.loss_fn.curriculum.update(trainer.current_epoch)
            pl_module.log(
                "hard_emphasis",
                pl_module.loss_fn.curriculum.hard_emphasis,
                on_step=False,
                on_epoch=True,
            )


class EMACallback(L.Callback):
    """
    Maintains an exponential moving average of the model's parameters and uses
    the EMA weights for validation/testing and in saved checkpoints.

    The shadow weights are updated after every optimizer step:
    `shadow = decay * shadow + (1 - decay) * param`. During validation the EMA
    weights are swapped into `pl_module.model` and the raw training weights are
    restored afterwards. `on_save_checkpoint` replaces the model weights in the
    checkpoint with the EMA weights (so checkpoints hold EMA weights; resuming
    restarts from them).
    """

    def __init__(self, decay: float):
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), but got {decay}")
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] | None = None
        self._backup: dict[str, torch.Tensor] | None = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        params = dict(pl_module.model.named_parameters())
        if self._shadow is None:
            self._shadow = {k: p.detach().clone() for k, p in params.items()}
        else:  # loaded from a checkpoint: move to the model's device
            self._shadow = {k: v.to(params[k].device) for k, v in self._shadow.items()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for k, p in pl_module.model.named_parameters():
                self._shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self._shadow is None:  # e.g. pre-train sanity check
            return
        with torch.no_grad():
            self._backup = {
                k: p.detach().clone() for k, p in pl_module.model.named_parameters()
            }
            for k, p in pl_module.model.named_parameters():
                p.copy_(self._shadow[k])

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self._backup is None:
            return
        with torch.no_grad():
            for k, p in pl_module.model.named_parameters():
                p.copy_(self._backup[k])
        self._backup = None

    on_test_start = on_validation_start
    on_test_end = on_validation_end

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self._shadow is None:
            return
        for k, v in self._shadow.items():
            checkpoint["state_dict"][f"model.{k}"] = v.clone()

    def state_dict(self):
        return {"shadow": self._shadow}

    def load_state_dict(self, state_dict):
        self._shadow = state_dict["shadow"]


class NoiseCurriculumCallback(L.Callback):
    """
    3-stage noise curriculum driving a `StreamingDecodingDataset`'s error rate:
    (1) hold at `p_start` for `stage1_epochs`, (2) anneal linearly to `p_end`
    over `anneal_epochs`, (3) hold at `p_end` until the end of training.

    Expects `trainer.datamodule.train_ds` to expose a settable `error_rate`
    (changes take effect at epoch boundaries: DataLoader workers re-copy the
    dataset each epoch with `persistent_workers=False`).
    """

    def __init__(
        self, *, p_start: float, p_end: float, stage1_epochs: int, anneal_epochs: int
    ):
        super().__init__()
        if p_start <= 0 or p_end <= 0:
            raise ValueError(
                f"Error rates must be positive, but got p_start={p_start}, p_end={p_end}"
            )
        if stage1_epochs < 0 or anneal_epochs < 0:
            raise ValueError(
                f"Epoch counts must be non-negative, but got "
                f"stage1_epochs={stage1_epochs}, anneal_epochs={anneal_epochs}"
            )
        self.p_start = p_start
        self.p_end = p_end
        self.stage1_epochs = stage1_epochs
        self.anneal_epochs = anneal_epochs

    def error_rate_at(self, epoch: int) -> float:
        if epoch < self.stage1_epochs:
            return self.p_start
        if epoch >= self.stage1_epochs + self.anneal_epochs:
            return self.p_end
        t = (epoch - self.stage1_epochs + 1) / (self.anneal_epochs + 1)
        return self.p_start + t * (self.p_end - self.p_start)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        p = self.error_rate_at(trainer.current_epoch)
        trainer.datamodule.train_ds.error_rate = p
        pl_module.print(f">>>>>> Noise curriculum: train error_rate = {p:.5f}")
        pl_module.log("train_error_rate", p, on_step=False, on_epoch=True)
