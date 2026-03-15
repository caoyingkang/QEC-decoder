"""Custom callbacks for Lightning."""

import logging

from lightning.pytorch.callbacks import Callback

class EpochSummary(Callback):
    """Prints a summary at the end of each validation epoch."""

    def on_train_epoch_end(self, trainer, pl_module):
        logging.debug("EpochSummary called...")
        print(f"\n--- Epoch {trainer.current_epoch} Summary ---")
        print(f"Train Loss: {trainer.callback_metrics['train_loss']:.5f}")
        print(f"Val Loss: {trainer.callback_metrics['val_loss']:.5f}")
        print("Val Metrics:")
        print(f"  Convergence Rate: {trainer.callback_metrics['convergence_rate'] * 100:.2f}%")
        print(f"  Logical Success Rate: {trainer.callback_metrics['logical_success_rate'] * 100:.2f}%")
        print(f"  Strict Success Rate: {trainer.callback_metrics['strict_success_rate'] * 100:.2f}%")
        print(f"  Accidental Success Rate: {trainer.callback_metrics['accidental_success_rate'] * 100:.2f}%")
        print(f"  Success Rate on Convergence: {trainer.callback_metrics['success_rate_on_convergence'] * 100:.2f}%")
        print(f"  Average Iterations: {trainer.callback_metrics['avg_iters']:.2f}")
        print(f"  Average Iterations on Convergence: {trainer.callback_metrics['avg_iters_on_convergence']:.2f}")
        print()
