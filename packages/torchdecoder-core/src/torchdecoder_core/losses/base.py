"""Base class for loss functions used to train iterative QEC decoders."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn


class LossResult(NamedTuple):
    """
    Return type of all decoding loss functions: a named tuple with fields `loss`,
    `synd_loss`, `obser_loss` (all float scalar tensors).
    """

    loss: torch.Tensor
    synd_loss: torch.Tensor
    obser_loss: torch.Tensor


class DecodingLoss(nn.Module, ABC):
    """
    A `torch.nn.Module` that serves as the base class for loss functions used to
    train iterative QEC decoders. Every decoding loss function should inherit from
    this class.

    The loss function is a weighted combination of two parts:
    `loss = beta * synd_loss + (1 - beta) * obser_loss`, where:
    - `synd_loss` measures how the decoder fails to match the syndromes.
    - `obser_loss` measures how the decoder fails to predict the logical observables.

    The weight `beta` ∈ [0, 1] controls the relative importance of the two parts.
    """

    def __init__(self, chkmat: np.ndarray, obsmat: np.ndarray, *, beta: float):
        """
        Initialize the decoding loss function.

        Parameters
        ----------
            chkmat : ndarray
                Check matrix, shape=(num_chks, num_vars), integer ∈ {0,1} or bool

            obsmat : ndarray
                Observable matrix, shape=(num_obsers, num_vars), integer ∈ {0,1} or bool

            beta : float
                Balance weight in [0, 1]. beta=1 → syndrome loss only; beta=0 → observable loss only.

        Notes
        -----
        The following tensors will be registered as `torch.nn.Module` buffers:

        - `chk_supp`: (num_chks, max_chk_weight), torch.long, semantics:
        `chk_supp[i, k]` = index of the `k`-th variable in the support of the `i`-th check (padded with 0).

        - `chk_mask`: (num_chks, max_chk_weight), torch.bool, semantics:
        `chk_mask[i, k]` = True if the `i`-th check involves at least `k` variables, False otherwise.

        - `obs_supp`: (num_obsers, max_obs_weight), torch.long, semantics:
        `obs_supp[i, k]` = index of the `k`-th variable in the support of the `i`-th observable (padded with 0).

        - `obs_mask`: (num_obsers, max_obs_weight), torch.bool, semantics:
        `obs_mask[i, k]` = True if the `i`-th observable involves at least `k` variables, False otherwise.
        """
        super().__init__()
        if not (0 <= beta <= 1):
            raise ValueError(f"beta must be in [0, 1], but got {beta}")
        self.beta = beta

        num_chks = chkmat.shape[0]
        num_obsers = obsmat.shape[0]

        max_chk_weight = int(chkmat.sum(axis=1).max())
        chk_supp = np.zeros((num_chks, max_chk_weight), dtype=np.int64)
        chk_mask = np.zeros((num_chks, max_chk_weight), dtype=bool)
        for i in range(num_chks):
            indices = np.nonzero(chkmat[i, :])[0]
            for k in range(len(indices)):
                chk_supp[i, k] = indices[k]
                chk_mask[i, k] = True

        max_obs_weight = int(obsmat.sum(axis=1).max())
        obs_supp = np.zeros((num_obsers, max_obs_weight), dtype=np.int64)
        obs_mask = np.zeros((num_obsers, max_obs_weight), dtype=bool)
        for i in range(num_obsers):
            indices = np.nonzero(obsmat[i, :])[0]
            for k in range(len(indices)):
                obs_supp[i, k] = indices[k]
                obs_mask[i, k] = True

        self.register_buffer(
            "chk_supp", torch.tensor(chk_supp, dtype=torch.long), persistent=False
        )  # (num_chks, max_chk_weight)
        self.register_buffer(
            "chk_mask", torch.tensor(chk_mask, dtype=torch.bool), persistent=False
        )  # (num_chks, max_chk_weight)
        self.register_buffer(
            "obs_supp", torch.tensor(obs_supp, dtype=torch.long), persistent=False
        )  # (num_obsers, max_obs_weight)
        self.register_buffer(
            "obs_mask", torch.tensor(obs_mask, dtype=torch.bool), persistent=False
        )  # (num_obsers, max_obs_weight)

    @abstractmethod
    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor,
    ) -> LossResult:
        """
        Parameters
        ----------
            llrs : torch.Tensor
                LLR outputs at all iterations, shape=(num_iters, batch_size, num_vars), float

            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_chks), int ∈ {0,1}

            observables : torch.Tensor
                Observable bits, shape=(batch_size, num_obsers), int ∈ {0,1}

        Returns
        -------
            LossResult
                Named tuple with fields `loss`, `synd_loss`, `obser_loss` (all float scalar tensors).
        """
        ...
