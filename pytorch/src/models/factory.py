import numpy as np
import torch.nn as nn
from omegaconf import DictConfig

from .learned_dmembp import LearnedDMemBP
from .multi_dmembp import MultiDMemBP


ACTIVATION_MAP = {
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}


def _resolve_activation(name: str) -> type[nn.Module]:
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Invalid activation: {name!r}, expected one of {list(ACTIVATION_MAP.keys())}")
    return ACTIVATION_MAP[name]


def build_decoder_model(chkmat: np.ndarray, prior: np.ndarray, model_cfg: DictConfig) -> nn.Module:
    model_name = model_cfg.name
    if model_name == "LearnedDMemBP":
        return LearnedDMemBP(
            chkmat, prior,
            num_iters=model_cfg.num_iters,
            min_impl_method=model_cfg.min_impl_method,
            sign_impl_method=model_cfg.sign_impl_method,
        )
    elif model_name == "MultiDMemBP":
        return MultiDMemBP(
            chkmat, prior,
            num_iters=model_cfg.num_iters,
            msg_features=model_cfg.msg_features,
            mlp_hidden_features=model_cfg.mlp.hidden_features,
            mlp_hidden_depth=model_cfg.mlp.hidden_depth,
            mlp_activation=_resolve_activation(model_cfg.mlp.activation),
            mlp_norm=model_cfg.mlp.norm,
            mlp_dropout_p=model_cfg.mlp.dropout_p,
            min_impl_method=model_cfg.min_impl_method,
            sign_impl_method=model_cfg.sign_impl_method,
            gamma_shared=model_cfg.gamma_shared,
            gamma_init=model_cfg.gamma_init,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name!r}")
