"""Base class for PyTorch iterative decoder models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DecoderModel(nn.Module, ABC):
    """
    Abstract base class for PyTorch iterative decoder models.

    This is the `torch.nn.Module` that implements the neural network architecture. Every decoder model
    should inherit from this class, and is assumed to have an iterative (recurrent) nature: the
    forward pass is made of multiple iterations, with each iteration producing an LLR value for
    each variable node.
    """

    def __init__(self, pcm: np.ndarray, prior: np.ndarray, num_iters: int):
        """Initialize the PyTorch iterative decoder model."""
        super().__init__()
        self.pcm = pcm
        self.prior = prior
        self.num_chks: int = pcm.shape[0]
        self.num_vars: int = pcm.shape[1]
        if num_iters < 1:
            raise ValueError(f"num_iters must be at least 1, but got {num_iters}")
        self.num_iters = num_iters

    @abstractmethod
    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Subclasses must implement this method.

        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_chks), int ∈ {0,1}

        Returns
        -------
            llrs : torch.Tensor
                LLR outputs at all iterations, shape=(num_iters, batch_size, num_vars), float
        """
        ...

    def load_lightning_checkpoint(
        self, ckpt_path: Path, skip_keys: list[str] = []
    ) -> None:
        """
        Load parameters and buffers from a Lightning checkpoint. Expect a checkpoint
        saved by a `LightningModule`, with `state_dict` keys prefixed by `"model."`.

        Parameters
        ----------
        ckpt_path : Path
            Path to the Lightning checkpoint file.

        skip_keys : list[str]
            List of keys (without prefix) to skip loading.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.

        RuntimeError
            If the checkpoint state_dict keys do not exactly match this model.
        """
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
