import numpy as np
import torch
import torch.nn.functional as F

from .base import DecodingLoss

EPS = 1e-6


# Shape hints code:
# B = batch_size
# C = num_chks
# O = num_obsers
# V = num_vars
# Wc = max_chk_weight
# Wo = max_obs_weight
# I = num_iters - skip_iters


class UniformIterationLoss(DecodingLoss):
    """
    A PyTorch `nn.Module` that implements a loss function for training iterative QEC decoders.

    Given inputs `llrs`, `syndromes`, and `observables`, the loss function consists of two parts:
    1. The first part measures how the decoder fails to match the syndromes.
    2. The second part measures how the decoder fails to predict the logical observables.

    For each iteration of each shot, the loss is a weighted sum of the above two parts controlled by a 
    hyperparameter `beta` ∈ [0,1]: `loss = beta * synd_loss + (1 - beta) * obser_loss`, where:
    - `synd_loss = mean(BCEWithLogitsLoss(-synd_pred_llr[i], syndromes[i]) for 0 <= i < num_chks)`, where 
    `synd_pred_llr[i]` is the LLR of the `i`-th syndrome bit obtained by XORing the error bits corresponding 
    to the `i`-th row of the check matrix.
    - `obser_loss = mean(BCEWithLogitsLoss(-obser_pred_llr[i], observables[i]) for 0 <= i < num_obsers)`, 
    where `obser_pred_llr[i]` is the LLR of the `i`-th observable bit obtained by XORing the error bits 
    corresponding to the `i`-th row of the observable matrix.

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
        super().__init__(chkmat, obsmat)
        if not (0 <= beta <= 1):
            raise ValueError(f"beta must be in [0, 1], but got {beta}")
        if skip_iters < 0:
            raise ValueError(f"skip_iters must be non-negative, but got {skip_iters}")
        self.beta = beta
        self.skip_iters = skip_iters

    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor,
    ) -> torch.Tensor:
        if self.skip_iters > 0:
            if self.skip_iters >= llrs.shape[0]:
                raise ValueError(f"skip_iters ({self.skip_iters}) must be less than num_iters ({llrs.shape[0]})")
            llrs = llrs[self.skip_iters:, :, :]

        # Recall: llr = log(Pr(E=0) / Pr(E=1)), so tanh(llr/2) = Pr(E=0) - Pr(E=1)
        tanhhalfllrs = torch.tanh(llrs * 0.5)  # (I, B, V)
        synd_loss = self._get_syndrome_loss(tanhhalfllrs, syndromes)
        obser_loss = self._get_observable_loss(tanhhalfllrs, observables)
        return self.beta * synd_loss + (1 - self.beta) * obser_loss

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

        # Recall: llr = log(Pr(E=0) / Pr(E=1)), so Sigmoid(-llr) = Pr(E=1), hence
        # binary_cross_entropy_with_logits(-llr, syndrome) = -log(Pr(E=syndrome))
        return F.binary_cross_entropy_with_logits(
            -synd_pred_llr,
            syndromes.float().unsqueeze(0).expand_as(synd_pred_llr),
            reduction="none",
        ).mean()

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
        ).mean()
