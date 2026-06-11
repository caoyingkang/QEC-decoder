"""Base class for PyTorch logical decoder models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .checkpoint_mixin import LightningCheckpointMixin


class LogicalDecoderModel(LightningCheckpointMixin, nn.Module, ABC):
    """
    Abstract base class for PyTorch logical decoder models.

    Unlike `DecoderModel` (iterative decoders that estimate an error pattern `ehat`),
    a logical decoder model predicts the logical observables directly: the forward
    pass maps a syndrome to one logit per logical observable. There is no
    `decode_inference` / `ehat` contract and no notion of convergence.
    """

    def __init__(self, num_chks: int, num_obsers: int):
        """
        Initialize the PyTorch logical decoder model.

        Parameters
        ----------
            num_chks : int
                Number of syndrome bits (detectors) in the input.

            num_obsers : int
                Number of logical observables to predict.
        """
        super().__init__()
        if num_chks < 1:
            raise ValueError(f"num_chks must be at least 1, but got {num_chks}")
        if num_obsers < 1:
            raise ValueError(f"num_obsers must be at least 1, but got {num_obsers}")
        self.num_chks = num_chks
        self.num_obsers = num_obsers

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
            logits : torch.Tensor
                Logical-observable logits, shape=(batch_size, num_obsers), float.
                Convention: `sigmoid(logit) = Pr(observable flipped)`.
        """
        ...
