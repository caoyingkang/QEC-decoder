"""Custom Lightning callbacks for training QEC decoders."""

import lightning as L


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
