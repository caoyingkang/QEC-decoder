import numpy as np
import torch

from .base import DecodingLoss, LossResult
from ..utils.decoding_utils import diagnose_convergence
from ..utils.tensor_utils import focal_BCE_with_logits

EPS = 1e-6

# Shape hints:
# B = batch_size
# C = num_chks
# O = num_obsers
# V = num_vars
# Wc = max_chk_weight
# Wo = max_obs_weight
# I = num_iters


class ConvergenceAwareLoss(DecodingLoss):
    """
    Convergence-aware loss function for iterative QEC decoders.

    In constrast to the `UniformIterationLoss`, this loss function is convergence-aware in the following sense:
    - For each shot, the syndrome loss is summed over all "active" iterations. These are the iterations 0, 1, ..., 
    output_iter, where output_iter is the first iteration where the hard-decision error pattern matches the 
    syndrome, or the last iteration if the syndrome is never matched.
    - For each shot, the observable loss is computed only at the output iteration.
    """

    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
        *,
        beta: float,
        focal_gamma: float,
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

            focal_gamma : float
                Focal loss exponent (>= 0). γ=0 disables focal modulation.
                Positive γ can help the decoder focus on hard-to-predict bits by introducing
                a (1 - p_t)^γ factor on the BCE loss, where p_t is the predicted probability 
                of the ground truth bit value.
        """
        super().__init__(chkmat, obsmat, beta=beta)
        if focal_gamma < 0:
            raise ValueError(f"focal_gamma must be non-negative, but got {focal_gamma}")
        self.num_vars = chkmat.shape[1]
        self.focal_gamma = focal_gamma

        # Register the check matrix as a buffer, used for convergence detection.
        self.register_buffer("chkmat", torch.tensor(chkmat, dtype=torch.float32), persistent=False)  # (num_chks, num_vars)

    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor,
    ) -> LossResult:
        num_iters = llrs.size(0)
        device = llrs.device

        # --- Convergence detection: Identify active iterations (non-differentiable) ---
        hard_decisions = (llrs < 0).float()  # (I, B, V), float ∈ {0.0, 1.0}
        _, output_iters = diagnose_convergence(hard_decisions, syndromes, self.chkmat)  # (B,), long
        iter_indices = torch.arange(num_iters, device=device)  # (I,), long
        active_iters_mask = iter_indices.unsqueeze(1) <= output_iters.unsqueeze(0)  # (I, B), bool

        tanhhalfllrs = torch.tanh(llrs * 0.5)  # (I, B, V)
        synd_loss = self._get_syndrome_loss(tanhhalfllrs, syndromes, active_iters_mask)
        obser_loss = self._get_observable_loss(tanhhalfllrs, observables, output_iters)
        loss = self.beta * synd_loss + (1.0 - self.beta) * obser_loss
        return LossResult(loss=loss, synd_loss=synd_loss, obser_loss=obser_loss)

    def _get_syndrome_loss(
        self,
        tanhhalfllrs: torch.Tensor,  # (I, B, V), float
        syndromes: torch.Tensor,  # (B, C), int ∈ {0,1}
        active_iters_mask: torch.Tensor,  # (I, B), bool
    ) -> torch.Tensor:
        """
        Syndrome loss: sum over active iterations, mean over batch, mean over checks.
        """
        # Gather LLRs of variables in the support of each check.
        gathered = tanhhalfllrs[:, :, self.chk_supp]  # (I, B, C, Wc)
        chk_mask_4d = self.chk_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, C, Wc)
        synd_pred_llr = 2.0 * (
            gathered.masked_fill(~chk_mask_4d, 1.0)
            .prod(dim=-1)
            .clamp(min=-1 + EPS, max=1 - EPS)
            .atanh()
        )  # (I, B, C)

        loss_per_check = focal_BCE_with_logits(
            -synd_pred_llr,
            syndromes.float().unsqueeze(0).expand_as(synd_pred_llr),
            gamma=self.focal_gamma,
        )  # (I, B, C)
        loss_per_iter = loss_per_check.mean(dim=-1)  # (I, B)
        loss_per_shot = torch.sum(
            loss_per_iter * active_iters_mask.float(),
            dim=0
        )  # (B,)
        return loss_per_shot.mean()

    def _get_observable_loss(
        self,
        tanhhalfllrs: torch.Tensor,  # (I, B, V), float
        observables: torch.Tensor,  # (B, O), int ∈ {0,1}
        output_iters: torch.Tensor,  # (B,), long
    ) -> torch.Tensor:
        """
        Observable loss: at output iteration only, mean over batch, mean over observables.
        """
        # Gather LLRs at the output iteration.
        index = output_iters.reshape(1, -1, 1).expand(1, -1, self.num_vars)  # (1, B, V)
        tanhhalfllrs_at_output = torch.gather(tanhhalfllrs, dim=0, index=index).squeeze(0)  # (B, V)

        # Gather LLRs of variables in the support of each observable.
        gathered = tanhhalfllrs_at_output[:, self.obs_supp]  # (B, O, Wo)
        obs_mask_3d = self.obs_mask.unsqueeze(0)  # (1, O, Wo)
        obser_pred_llr = 2.0 * (
            gathered.masked_fill(~obs_mask_3d, 1.0)
            .prod(dim=-1)
            .clamp(min=-1 + EPS, max=1 - EPS)
            .atanh()
        )  # (B, O)

        return focal_BCE_with_logits(
            -obser_pred_llr,
            observables.float(),
            gamma=self.focal_gamma,
        ).mean()
