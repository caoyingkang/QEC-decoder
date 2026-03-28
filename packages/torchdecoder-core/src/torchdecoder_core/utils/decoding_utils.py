"""Utilities for LLR processing in iterative decoders."""

import torch

from .tensor_utils import matmul_GF2


def diagnose_convergence(
    hard_decisions: torch.Tensor,
    syndromes: torch.Tensor,
    chkmat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Diagnose whether the decoder converges in each shot.

    Parameters
    ----------
    hard_decisions : torch.Tensor
        Hard decisions from all iterations, shape=(num_iters, batch_size, num_vars),
        float ∈ {0.0, 1.0}.

    syndromes : torch.Tensor
        Syndrome bits, shape=(batch_size, num_chks), int ∈ {0, 1}.

    chkmat : torch.Tensor
        Check matrix, shape=(num_chks, num_vars), float ∈ {0.0, 1.0}.

    Returns
    -------
    converged_mask : torch.Tensor
        Boolean mask, shape=(batch_size,), True if syndrome matched at some iteration.

    output_iters : torch.Tensor
        Output iteration index for each shot, shape=(batch_size,), long. For converged shots,
        this is the first iteration where the syndrome is matched; otherwise num_iters - 1.
    """
    num_iters = hard_decisions.size(0)
    synd_pred = matmul_GF2(hard_decisions, chkmat.T)  # (I, B, C), int ∈ {0, 1}
    synd_matched_mask = torch.all(
        synd_pred == syndromes.unsqueeze(0), dim=2
    )  # (I, B), bool
    converged_mask = torch.any(synd_matched_mask, dim=0)  # (B,), bool
    output_iters = torch.where(
        converged_mask,
        synd_matched_mask.int().argmax(dim=0),
        num_iters - 1,
    )  # (B,), long
    return converged_mask, output_iters


def gather_ehat(
    hard_decisions: torch.Tensor, output_iters: torch.Tensor
) -> torch.Tensor:
    """
    Gather the estimated error pattern at the output iteration for each shot.

    Parameters
    ----------
    hard_decisions : torch.Tensor
        Hard decisions from all iterations, shape=(num_iters, batch_size, num_vars),
        float ∈ {0.0, 1.0}.

    output_iters : torch.Tensor
        Output iteration index for each shot, shape=(batch_size,), long. For converged shots,
        this is the first iteration where the syndrome is matched; otherwise num_iters - 1.

    Returns
    -------
    ehat : torch.Tensor
        Estimated error pattern, shape=(batch_size, num_vars), float ∈ {0.0, 1.0}.
    """
    num_vars = hard_decisions.size(2)
    index = output_iters.reshape(1, -1, 1).expand(1, -1, num_vars)  # (1, B, V), long
    ehat = torch.gather(hard_decisions, dim=0, index=index).squeeze(
        0
    )  # (B, V), float ∈ {0.0, 1.0}
    return ehat
