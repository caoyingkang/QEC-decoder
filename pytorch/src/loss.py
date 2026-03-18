import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# Shape hints code:
# B = batch_size
# C = num_chks
# O = num_obsers
# V = num_vars
# Wc = max_chk_weight
# Wo = max_obs_weight
# I = num_iters - skip_iters


class IterativeDecodingLoss(nn.Module):
    """
    A PyTorch `nn.Module` that implements a loss function for training iterative QEC decoders.

    Given inputs `llrs`, `syndromes`, and `observables`, the loss function consists of two parts:
    1. The first part measures how the decoder fails to match the syndromes.
    2. The second part measures how the decoder fails to predict the logical observables.

    For each iteration of each shot, the loss is a weighted sum of the above two parts controlled by a 
    hyperparameter `beta` ∈ [0,1]: `loss = beta * sum(loss_synd) + (1 - beta) * sum(loss_obser)`, where:
    - `loss_synd[i] = BCEWithLogitsLoss(-synd_pred_llr[i], syndromes[i])`, where `synd_pred_llr[i]` is the
    LLR of the `i`-th syndrome bit obtained by XORing the error bits corresponding to the `i`-th row of the
    check matrix `chkmat`.
    - `loss_obser[i] = BCEWithLogitsLoss(-obser_pred_llr[i], observables[i])`, where `obser_pred_llr[i]` is the
    LLR of the `i`-th observable bit obtained by XORing the error bits corresponding to the `i`-th row of the
    observable matrix `obsmat`.

    To calculate the total loss for a batch, we average the loss over the iterations and the shots.
    """

    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
        *,
        beta: float,
        skip_iters: int,
    ):
        """
        Parameters
        ----------
            chkmat : ndarray
                Check matrix, shape=(num_chks, num_vars), integer ∈ {0,1} or bool

            obsmat : ndarray
                Observable matrix, shape=(num_obsers, num_vars), integer ∈ {0,1} or bool

            beta : float
                Hyperparameter that balances the contribution of the two parts of the loss function.
                If beta = 1.0, then only the part corresponding to syndrome matching is included.
                If beta = 0.0, then only the part corresponding to observable prediction is included.

            skip_iters : int
                The first `skip_iters` iterations are skipped in the calculation of the loss.
                Default is 0, meaning that the LLRs output from all iterations contribute to the loss.
        """
        super().__init__()
        if not (0 <= beta <= 1):
            raise ValueError(f"beta must be in [0, 1], but got {beta}")
        if skip_iters < 0:
            raise ValueError(f"skip_iters must be non-negative, but got {skip_iters}")

        num_chks = chkmat.shape[0]
        num_obsers = obsmat.shape[0]
        self.beta = beta
        self.skip_iters = skip_iters

        # Build padded check → variable table.
        # chk_supp[i, k] = index of the k-th variable in the support of the i-th check (padded with 0).
        # chk_mask[i, k] = True if the i-th check involves at least k variables, False otherwise.
        max_chk_weight = int(chkmat.sum(axis=1).max())
        chk_supp = np.zeros((num_chks, max_chk_weight), dtype=np.int64)
        chk_mask = np.zeros((num_chks, max_chk_weight), dtype=bool)
        for i in range(num_chks):
            indices = np.nonzero(chkmat[i, :])[0]
            for k in range(len(indices)):
                chk_supp[i, k] = indices[k]
                chk_mask[i, k] = True

        # Build padded observable → variable table.
        # obs_supp[i, k] = index of the k-th variable in the support of the i-th observable (padded with 0).
        # obs_mask[i, k] = True if the i-th observable involves at least k variables, False otherwise.
        max_obs_weight = int(obsmat.sum(axis=1).max())
        obs_supp = np.zeros((num_obsers, max_obs_weight), dtype=np.int64)
        obs_mask = np.zeros((num_obsers, max_obs_weight), dtype=bool)
        for i in range(num_obsers):
            indices = np.nonzero(obsmat[i, :])[0]
            for k in range(len(indices)):
                obs_supp[i, k] = indices[k]
                obs_mask[i, k] = True

        self.register_buffer("chk_supp", torch.tensor(chk_supp, dtype=torch.long), persistent=False)  # (C, Wc)
        self.register_buffer("chk_mask", torch.tensor(chk_mask, dtype=torch.bool), persistent=False)  # (C, Wc)
        self.register_buffer("obs_supp", torch.tensor(obs_supp, dtype=torch.long), persistent=False)  # (O, Wo)
        self.register_buffer("obs_mask", torch.tensor(obs_mask, dtype=torch.bool), persistent=False)  # (O, Wo)

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
        if self.skip_iters > 0:
            if self.skip_iters >= llrs.shape[0]:
                raise ValueError(f"skip_iters ({self.skip_iters}) must be less than num_iters ({llrs.shape[0]})")
            llrs = llrs[self.skip_iters:, :, :]

        tanhhalfllrs = torch.tanh(llrs / 2.)  # (I, B, V)
        loss_synd = self._get_syndrome_loss(tanhhalfllrs, syndromes)  # (I, B, C)
        loss_synd_sum = loss_synd.sum(dim=-1)  # (I, B)
        loss_obser = self._get_observable_loss(tanhhalfllrs, observables)  # (I, B, O)
        loss_obser_sum = loss_obser.sum(dim=-1)  # (I, B)
        loss = self.beta * loss_synd_sum.mean() + (1 - self.beta) * loss_obser_sum.mean()

        return loss

    def _get_syndrome_loss(
        self,
        tanhhalfllrs: torch.Tensor,  # (I, B, V), float
        syndromes: torch.Tensor,  # (B, C), int ∈ {0,1}
    ) -> torch.Tensor:
        # Gather LLRs of variables in the support of each check.
        gathered = tanhhalfllrs[:, :, self.chk_supp]  # (I, B, C, Wc)
        chk_mask_4d = self.chk_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, C, Wc)
        synd_pred_llr = 2.0 * (
            gathered.masked_fill(~chk_mask_4d, 1.0)
            .prod(dim=-1)
            .clamp(min=-1 + EPS, max=1 - EPS)
            .atanh()
        )  # (I, B, C)

        return F.binary_cross_entropy_with_logits(
            -synd_pred_llr,
            syndromes.float().unsqueeze(0).expand_as(synd_pred_llr),
            reduction="none",
        )  # (I, B, C)

    def _get_observable_loss(
        self,
        tanhhalfllrs: torch.Tensor,  # (I, B, V), float
        observables: torch.Tensor,  # (B, O), int ∈ {0,1}
    ) -> torch.Tensor:
        # Gather LLRs of variables in the support of each observable.
        gathered = tanhhalfllrs[:, :, self.obs_supp]  # (I, B, O, Wo)
        obs_mask_4d = self.obs_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, O, Wo)
        obser_pred_llr = 2.0 * (
            gathered.masked_fill(~obs_mask_4d, 1.0)
            .prod(dim=-1)
            .clamp(min=-1 + EPS, max=1 - EPS)
            .atanh()
        )  # (I, B, O)

        return F.binary_cross_entropy_with_logits(
            -obser_pred_llr,
            observables.float().unsqueeze(0).expand_as(obser_pred_llr),
            reduction="none",
        )  # (I, B, O)
