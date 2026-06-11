"""Mixin providing Lightning-checkpoint loading for decoder models."""

from pathlib import Path

import torch


class LightningCheckpointMixin:
    """
    Mixin for `torch.nn.Module` subclasses that adds `load_lightning_checkpoint`.

    Shared by `DecoderModel` (iterative decoders) and `LogicalDecoderModel`
    (logical decoders).
    """

    def load_lightning_checkpoint(
        self, ckpt_path: Path, skip_keys: list[str] | None = None
    ) -> None:
        """
        Load parameters and buffers from a Lightning checkpoint. Expect a checkpoint
        saved by a `LightningModule`, with `state_dict` keys prefixed by `"model."`.

        Parameters
        ----------
        ckpt_path : Path
            Path to the Lightning checkpoint file.

        skip_keys : list[str] | None
            List of keys (without prefix) to skip loading.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.

        RuntimeError
            If the checkpoint state_dict keys do not exactly match this model.
        """
        if skip_keys is None:
            skip_keys = []
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        prefix = "model."
        current_state_dict = self.state_dict()
        new_state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith(prefix):
                key = k[len(prefix) :]
                new_state_dict[key] = current_state_dict[key] if key in skip_keys else v

        self.load_state_dict(new_state_dict, strict=True)
