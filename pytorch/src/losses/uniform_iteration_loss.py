import numpy as np
import torch
import torch.nn.functional as F

from .base import DecodingLoss, LossResult

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
    A loss function for iterative QEC decoders that treats all iterations equally.

    For each shot, at each iteration, the loss is calculated as follows:
    - `synd_loss = mean(BCEWithLogitsLoss(-synd_pred_llr[i], syndromes[i]) for 0 <= i < num_chks)`, where 
    `synd_pred_llr[i]` is the LLR of the `i`-th syndrome bit obtained by XORing the error bits corresponding 
    to the `i`-th row of the check matrix.
    - `obser_loss = mean(BCEWithLogitsLoss(-obser_pred_llr[i], observables[i]) for 0 <= i < num_obsers)`, 
    where `obser_pred_llr[i]` is the LLR of the `i`-th observable bit obtained by XORing the error bits 
    corresponding to the `i`-th row of the observable matrix.

    To calculate the overall loss for a batch, we average the loss over the iterations and the shots.
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
                Balance weight in [0, 1]. beta=1 → syndrome loss only; beta=0 → observable loss only.

            skip_iters : int
                The first `skip_iters` iterations are skipped in the calculation of the loss.
                Default is 0, meaning that the LLRs output from all iterations contribute to the loss.
        """
        super().__init__(chkmat, obsmat, beta=beta)
        if skip_iters < 0:
            raise ValueError(f"skip_iters must be non-negative, but got {skip_iters}")
        self.skip_iters = skip_iters

    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor,
    ) -> LossResult:
        if self.skip_iters > 0:
            if self.skip_iters >= llrs.shape[0]:
                raise ValueError(f"skip_iters ({self.skip_iters}) must be less than num_iters ({llrs.shape[0]})")
            llrs = llrs[self.skip_iters:, :, :]

        # Recall: llr = log(Pr(E=0) / Pr(E=1)), so tanh(llr/2) = Pr(E=0) - Pr(E=1)
        tanhhalfllrs = torch.tanh(llrs * 0.5)  # (I, B, V)
        synd_loss = self._get_syndrome_loss(tanhhalfllrs, syndromes)
        obser_loss = self._get_observable_loss(tanhhalfllrs, observables)
        loss = self.beta * synd_loss + (1 - self.beta) * obser_loss
        return LossResult(loss=loss, synd_loss=synd_loss, obser_loss=obser_loss)

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
