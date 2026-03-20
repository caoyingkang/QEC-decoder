"""Base class for loss functions used to train iterative QEC decoders."""
import numpy as np
import torch
import torch.nn as nn


class DecodingLoss(nn.Module):
    """
    A `torch.nn.Module` that serves as the base class for loss functions used to 
    train iterative QEC decoders. Every decoding loss function should inherit from 
    this class.
    """

    def __init__(self, chkmat: np.ndarray, obsmat: np.ndarray):
        """
        Initialize the decoding loss function with the check matrix and the observable matrix.
        Register the following tensors as `torch.nn.Module` buffers:

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
        
        self.register_buffer("chk_supp", torch.tensor(chk_supp, dtype=torch.long), persistent=False)  # (num_chks, max_chk_weight)
        self.register_buffer("chk_mask", torch.tensor(chk_mask, dtype=torch.bool), persistent=False)  # (num_chks, max_chk_weight)
        self.register_buffer("obs_supp", torch.tensor(obs_supp, dtype=torch.long), persistent=False)  # (num_obsers, max_obs_weight)
        self.register_buffer("obs_mask", torch.tensor(obs_mask, dtype=torch.bool), persistent=False)  # (num_obsers, max_obs_weight)

    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor,
    ) -> torch.Tensor:
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
            loss : torch.Tensor
                Scalar loss, float
        """
        raise NotImplementedError("Subclasses must implement this method.")
