"""Factory for building loss functions."""

import numpy as np
from omegaconf import DictConfig

from .base import DecodingLoss
from .uniform_iteration_loss import UniformIterationLoss
from .convergence_aware_loss import ConvergenceAwareLoss


def build_decoding_loss(
    chkmat: np.ndarray,
    obsmat: np.ndarray,
    loss_cfg: DictConfig,
) -> DecodingLoss:
    """
    Given a check matrix, an observable matrix, and a configuration, build a loss function
    for training iterative QEC decoders.
    """
    loss_name = loss_cfg.name
    if loss_name == "UniformIterationLoss":
        return UniformIterationLoss(
            chkmat,
            obsmat,
            beta=loss_cfg.beta,
            skip_iters=loss_cfg.skip_iters,
        )
    elif loss_name == "ConvergenceAwareLoss":
        return ConvergenceAwareLoss(
            chkmat,
            obsmat,
            beta=loss_cfg.beta,
            focal_gamma=loss_cfg.focal_gamma,
        )
    else:
        raise ValueError(f"Invalid loss function name: {loss_name!r}")
