"""Utilities for LLR processing in iterative decoders."""
import torch

from .tensor_utils import INT_DTYPE, FLOAT_DTYPE


def llrs_to_ehat(
    llrs: torch.Tensor,
    syndromes: torch.Tensor,
    chkmat: torch.Tensor,
) -> torch.Tensor:
    """
    Convert LLR output from iterative decoder to final error predictions.

    Use syndrome-matched early stopping: if the decoder converges (syndrome
    matches) at an earlier iteration, use that; otherwise use the last iteration.

    Assume that the input tensors are on the same device.

    Parameters
    ----------
    llrs : torch.Tensor
        LLR values at all iterations, shape=(num_iters, batch_size, num_vars), float.

    syndromes : torch.Tensor
        Syndrome bits, shape=(batch_size, num_chks), int.

    chkmat : torch.Tensor
        Check matrix, shape=(num_chks, num_vars), int if device is CPU, float if device is GPU.

    Returns
    -------
    ehat : torch.Tensor
        Decoded error pattern, shape=(batch_size, num_vars), int ∈ {0,1}.

    converged_mask : torch.Tensor
        Boolean mask, shape=(batch_size,), True if syndrome matched at some iteration.
    """
    num_iters, batch_size, num_vars = llrs.shape

    if llrs.is_cpu:
        hard_decisions = (llrs < 0).to(INT_DTYPE)  # (num_iters, B, V), int
        synd_pred = torch.matmul(hard_decisions, chkmat.T) % 2  # (num_iters, B, C), int
    else:
        # cuda does not support matrix multiplication for integer tensors
        hard_decisions = (llrs < 0).to(FLOAT_DTYPE)  # (num_iters, B, V), float
        synd_pred_raw = torch.matmul(hard_decisions, chkmat.T)  # (num_iters, B, C), float
        synd_pred = torch.round(synd_pred_raw).to(INT_DTYPE) % 2  # (num_iters, B, C), int

    synd_matched_mask = torch.all(synd_pred == syndromes.unsqueeze(0), dim=2)  # (num_iters, B), bool
    converged_mask = torch.any(synd_matched_mask, dim=0)  # (B,), bool

    # For each shot, find which iteration is the overall output of the decoder:
    # If the decoder converges, this is the first iteration where the syndrome is matched;
    # If the decoder does not converge, this is the last iteration.
    output_iters = torch.where(
        converged_mask,
        synd_matched_mask.int().argmax(dim=0),
        num_iters - 1,
    )  # (B,), int

    # Get the output error pattern for each shot
    index = output_iters.reshape(1, batch_size, 1).expand(1, batch_size, num_vars)  # (1, B, V), int
    ehat = torch.gather(hard_decisions, dim=0, index=index).squeeze(0)  # (B, V), int
    return ehat, converged_mask
