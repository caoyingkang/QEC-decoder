"""Factory for building loss functions."""

import numpy as np
from omegaconf import DictConfig
import torch.nn as nn

from .convergence_aware_loss import ConvergenceAwareLoss
from .curriculum import Curriculum
from .logical_bce_loss import LogicalBCELoss
from .uniform_iteration_loss import UniformIterationLoss


def build_loss_fn(
    chkmat: np.ndarray, obsmat: np.ndarray, loss_cfg: DictConfig
) -> nn.Module:
    """
    Given a check matrix, an observable matrix, and a configuration, build a loss
    function for training QEC decoder models. Covers both iterative decoding losses
    (which need `chkmat`/`obsmat`) and logical losses (which ignore them).
    """
    match loss_cfg.name:
        case "UniformIterationLoss":
            return UniformIterationLoss(
                chkmat,
                obsmat,
                beta=loss_cfg.beta,
                skip_iters=loss_cfg.skip_iters,
            )
        case "ConvergenceAwareLoss":
            if "curriculum" in loss_cfg:
                curriculum = Curriculum(
                    max_emphasis=loss_cfg.curriculum.max_emphasis,
                    ramp_epochs=loss_cfg.curriculum.ramp_epochs,
                )
            else:
                curriculum = None
            return ConvergenceAwareLoss(
                chkmat,
                obsmat,
                beta=loss_cfg.beta,
                focal_gamma=loss_cfg.focal_gamma,
                curriculum=curriculum,
            )
        case "LogicalBCELoss":
            return LogicalBCELoss()
        case _:
            raise ValueError(f"Invalid loss function name: {loss_cfg.name!r}")
