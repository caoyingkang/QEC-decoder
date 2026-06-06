"""Factory for building PyTorch decoder models."""

import numpy as np
from omegaconf import DictConfig

from .base import DecoderModel
from .learned_dmembp import LearnedDMemBP
from .multi_dmembp import MultiDMemBP


def build_decoder_model(
    chkmat: np.ndarray, prior: np.ndarray, model_cfg: DictConfig
) -> DecoderModel:
    """
    Build a PyTorch decoder model for a check matrix and a prior error probability vector, according to a model configuration.
    """
    match model_cfg.name:
        case "LearnedDMemBP":
            return LearnedDMemBP(
                chkmat,
                prior,
                model_cfg.num_iters,
                min_impl_method=model_cfg.min_impl_method,
                sign_impl_method=model_cfg.sign_impl_method,
            )
        case "MultiDMemBP":
            return MultiDMemBP(
                chkmat,
                prior,
                model_cfg.num_iters,
                msg_features=model_cfg.msg_features,
                mlp_hidden_features=model_cfg.mlp.hidden_features,
                mlp_hidden_depth=model_cfg.mlp.hidden_depth,
                mlp_activation=model_cfg.mlp.activation,
                mlp_norm=model_cfg.mlp.norm,
                mlp_dropout_p=model_cfg.mlp.dropout_p,
                min_impl_method=model_cfg.min_impl_method,
                sign_impl_method=model_cfg.sign_impl_method,
                gamma_shared=model_cfg.gamma_shared,
                gamma_init=model_cfg.gamma_init,
            )
        case _:
            raise ValueError(f"Invalid model name: {model_cfg.name!r}")
