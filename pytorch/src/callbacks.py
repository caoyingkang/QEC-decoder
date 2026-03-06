"""Custom callbacks for Lightning."""

import logging

from lightning.pytorch.callbacks import Callback

class EpochSummary(Callback):
    """Prints a summary at the end of each validation epoch."""

    def on_train_epoch_end(self, trainer, pl_module):
        logging.debug("EpochSummary called...")
        print(f"\n--- Epoch {trainer.current_epoch} Summary ---")
        print(f"Train Loss: {trainer.callback_metrics['train_loss']:.6f}")
        print(f"Val Loss: {trainer.callback_metrics['val_loss']:.6f}")
        print("Val Metrics:")
        print(f"  Wrong Syndrome Rate: {trainer.callback_metrics['wrong_syndrome_rate']:.6f}")
        print(f"  Wrong Observable Rate: {trainer.callback_metrics['wrong_observable_rate']:.6f}")
        print(f"  Wrong Either Rate: {trainer.callback_metrics['wrong_either_rate']:.6f}")
        print()
