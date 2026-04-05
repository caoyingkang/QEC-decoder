"""Factory for building PyTorch decoder models."""

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .base import DecoderModel
from .learned_dmembp import LearnedDMemBP
from .multi_dmembp import MultiDMemBP


def build_decoder_model(
    chkmat: np.ndarray, prior: np.ndarray, model_cfg: DictConfig
) -> DecoderModel:
    """
    Build a PyTorch decoder model for a check matrix and a prior error probability vector, according to a model configuration.
    """
    model_name = model_cfg.name
    if model_name == "LearnedDMemBP":
        return LearnedDMemBP(
            chkmat,
            prior,
            model_cfg.num_iters,
            min_impl_method=model_cfg.min_impl_method,
            sign_impl_method=model_cfg.sign_impl_method,
            use_edge_weights=OmegaConf.select(
                model_cfg, "use_edge_weights", default=False
            ),
        )
    elif model_name == "MultiDMemBP":
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
            llr_pooling=OmegaConf.select(model_cfg, "llr_pooling", default="mean"),
            use_edge_weights=OmegaConf.select(
                model_cfg, "use_edge_weights", default=False
            ),
        )
    else:
        raise ValueError(f"Invalid model name: {model_name!r}")
