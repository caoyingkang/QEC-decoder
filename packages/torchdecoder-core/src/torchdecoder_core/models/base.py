"""Base class for PyTorch iterative decoder models."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from ..utils.decoding_utils import diagnose_convergence, gather_ehat
from .checkpoint_mixin import LightningCheckpointMixin


class InferenceResult(NamedTuple):
    """Output of `DecoderModel.decode_inference`.

    Attributes
    ----------
    ehat : torch.Tensor
        Estimated error pattern, shape=(batch_size, num_vars), float ∈ {0.0, 1.0}
    converged_mask : torch.Tensor
        Boolean mask, shape=(batch_size,), True if syndrome matched.
    decoding_iters : torch.Tensor
        Number of iterations the decoder ran for each shot, shape=(batch_size,), long.
        For shots with trivial (all-zero) syndrome, this is 0, meaning no decoding was performed.
        For shots that did not converge, this is num_iters.
    """

    ehat: torch.Tensor
    converged_mask: torch.Tensor
    decoding_iters: torch.Tensor


class DecoderModel(LightningCheckpointMixin, nn.Module, ABC):
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

    def decode_inference(
        self, syndromes: torch.Tensor, chkmat: torch.Tensor
    ) -> InferenceResult:
        """
        Inference path: decode syndromes and return error estimate and convergence info.

        Default implementation runs the full `forward` then `diagnose_convergence` and
        `gather_ehat`. Subclasses may override with an early-terminating loop.

        Parameters
        ----------
        syndromes : torch.Tensor
            Syndrome bits, shape=(batch_size, num_chks), int ∈ {0,1}
        chkmat : torch.Tensor
            Parity-check matrix, shape=(num_chks, num_vars), float ∈ {0.0, 1.0}

        Returns
        -------
        InferenceResult
        """
        llrs = self.forward(syndromes)
        hard_decisions = (llrs < 0).float()
        converged_mask, output_iters = diagnose_convergence(
            hard_decisions, syndromes, chkmat
        )
        ehat = gather_ehat(hard_decisions, output_iters)
        decoding_iters = output_iters + 1

        trivial_mask = torch.all(syndromes == 0, dim=1)  # (B,), bool
        ehat[trivial_mask] = 0
        converged_mask |= trivial_mask
        decoding_iters[trivial_mask] = 0

        return InferenceResult(ehat, converged_mask, decoding_iters)
