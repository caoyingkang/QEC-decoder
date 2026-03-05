import numpy as np
import torch.nn as nn
from omegaconf import DictConfig

from .learned_dmembp import LearnedDMemBP


def build_decoder_model(chkmat: np.ndarray, prior: np.ndarray, model_cfg: DictConfig) -> nn.Module:
    model_name = model_cfg.name
    if model_name == "learned_dmembp":
        return LearnedDMemBP(
            chkmat, prior,
            num_iters=model_cfg.num_iters,
            min_impl_method=model_cfg.min_impl_method,
            sign_impl_method=model_cfg.sign_impl_method,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name!r}")
