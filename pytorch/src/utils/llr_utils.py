"""Utilities for LLR processing in iterative decoders."""
import torch

from .tensor_utils import matmul_GF2

def llrs_to_ehat(
    llrs: torch.Tensor,
    syndromes: torch.Tensor,
    chkmat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert LLR output from iterative decoder to final error predictions.

    Use syndrome-matched early stopping: if the decoder converges (syndrome
    matches) at an earlier iteration, use that; otherwise use the last iteration.

    Parameters
    ----------
    llrs : torch.Tensor
        LLR output from all iterations, shape=(num_iters, batch_size, num_vars), float.

    syndromes : torch.Tensor
        Syndrome bits, shape=(batch_size, num_chks), int ∈ {0, 1}.

    chkmat : torch.Tensor
        Check matrix, shape=(num_chks, num_vars), int ∈ {0, 1} or float ∈ {0.0, 1.0} (float is preferred).

    Returns
    -------
    ehat : torch.Tensor
        Decoded error pattern, shape=(batch_size, num_vars), float ∈ {0.0, 1.0}.

    converged_mask : torch.Tensor
        Boolean mask, shape=(batch_size,), True if syndrome matched at some iteration.

    output_iters : torch.Tensor
        Output iteration index for each shot, shape=(batch_size,), long. For converged shots,
        this is the first iteration where the syndrome is matched; otherwise num_iters - 1.
    """
    num_iters, batch_size, num_vars = llrs.shape

    hard_decisions = (llrs < 0).float()  # (I, B, V), float ∈ {0.0, 1.0}
    synd_pred = matmul_GF2(hard_decisions, chkmat.T)  # (I, B, C), int ∈ {0, 1}

    synd_matched_mask = torch.all(synd_pred == syndromes.unsqueeze(0), dim=2)  # (I, B), bool
    converged_mask = torch.any(synd_matched_mask, dim=0)  # (B,), bool

    # For each shot, find which iteration is the overall output of the decoder:
    # If the decoder converges, this is the first iteration where the syndrome is matched;
    # If the decoder does not converge, this is the last iteration.
    output_iters = torch.where(
        converged_mask,
        synd_matched_mask.int().argmax(dim=0),
        num_iters - 1,
    )  # (B,), long

    # Get the output error pattern for each shot
    index = output_iters.reshape(1, batch_size, 1).expand(1, batch_size, num_vars)  # (1, B, V), long
    ehat = torch.gather(hard_decisions, dim=0, index=index).squeeze(0)  # (B, V), float ∈ {0.0, 1.0}
    return ehat, converged_mask, output_iters
