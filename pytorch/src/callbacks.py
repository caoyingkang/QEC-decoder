"""Custom callbacks for Lightning."""

from lightning.pytorch.callbacks import Callback


class TrainingProgressSummary(Callback):
    """Print a summary of training progress."""

    def on_sanity_check_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            m = trainer.callback_metrics
            print("\n--- Pre-Train Validation Summary ---")
            print(f"Val Loss: {m['val_loss']:.5f}")
            print("Val Metrics:")
            print(f"  Convergence Rate: {m['convergence_rate'] * 100:.2f}%")
            print(f"  Logical Success Rate: {m['logical_success_rate'] * 100:.2f}%")
            print(f"  Strict Success Rate: {m['strict_success_rate'] * 100:.2f}%")
            print(f"  Accidental Success Rate: {m['accidental_success_rate'] * 100:.2f}%")
            print(f"  Success Rate on Convergence: {m['success_rate_on_convergence'] * 100:.2f}%")
            print(f"  Average Iterations: {m['avg_iters']:.2f}")
            print(f"  Average Iterations on Convergence: {m['avg_iters_on_convergence']:.2f}")
            print()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            m = trainer.callback_metrics
            pl_module.print(f"\n--- Epoch {trainer.current_epoch} Summary ---")
            pl_module.print(f"Train Loss: {m['train_loss']:.5f}")
            pl_module.print(f"Val Loss: {m['val_loss']:.5f}")
            pl_module.print("Val Metrics:")
            pl_module.print(f"  Convergence Rate: {m['convergence_rate'] * 100:.2f}%")
            pl_module.print(f"  Logical Success Rate: {m['logical_success_rate'] * 100:.2f}%")
            pl_module.print(f"  Strict Success Rate: {m['strict_success_rate'] * 100:.2f}%")
            pl_module.print(f"  Accidental Success Rate: {m['accidental_success_rate'] * 100:.2f}%")
            pl_module.print(f"  Success Rate on Convergence: {m['success_rate_on_convergence'] * 100:.2f}%")
            pl_module.print(f"  Average Iterations: {m['avg_iters']:.2f}")
            pl_module.print(f"  Average Iterations on Convergence: {m['avg_iters_on_convergence']:.2f}")
            pl_module.print()
