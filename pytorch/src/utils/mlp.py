"""Multi-layer perceptron network."""
from typing import Optional

import torch.nn as nn

from .tensor_utils import FLOAT_DTYPE

NORM_CLS_MAP = {
    "LayerNorm": nn.LayerNorm,
    "RMSNorm": nn.RMSNorm,
}


def _resolve_norm(name: str) -> type[nn.Module]:
    if name not in NORM_CLS_MAP:
        raise ValueError(f"Unsupported normalization: {name!r}, expected one of {list(NORM_CLS_MAP.keys())}")
    return NORM_CLS_MAP[name]


class MLP(nn.Module):
    """
    Multi-layer perceptron network.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_depth: int,
        activation: nn.Module,
        *,
        norm: Optional[str],
        dropout_p: Optional[float],
        zero_init: bool = False,
        residual: bool = False,
    ):
        """
        Parameters
        ----------
            in_features : int
                Number of input features.

            out_features : int
                Number of output features.

            hidden_features : int
                Number of features in each hidden layer.

            hidden_depth : int
                Number of hidden layers. Must be at least 1.

            activation : nn.Module
                Activation function to use in the hidden layers.

            norm : str | None
                Normalization to use in the hidden layers. If None, no normalization is applied.
                Allowed options are "LayerNorm" and "RMSNorm".

            dropout_p : float | None
                Dropout probability for the hidden layers. If None, no dropout is applied.

            zero_init : bool
                If True, initialize all linear layers with zero weights and biases.

            residual : bool
                If True, use residual form output = x + net(x). Requires in_features == out_features.

        Notes
        -----
        When `in_features == out_features`, one can set `zero_init=True` and `residual=True` to initialize 
        the MLP as an identity function.
        """
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be at least 1, but got {in_features}")
        if out_features < 1:
            raise ValueError(f"out_features must be at least 1, but got {out_features}")
        if hidden_features < 1:
            raise ValueError(f"hidden_features must be at least 1, but got {hidden_features}")
        if hidden_depth < 1:
            raise ValueError(f"hidden_depth must be at least 1, but got {hidden_depth}")
        if dropout_p is not None and (dropout_p < 0 or dropout_p > 1):
            raise ValueError(f"dropout_p must be None or a number between 0 and 1, but got {dropout_p}")
        if residual and in_features != out_features:
            raise ValueError("Cannot set residual to True when in_features != out_features")
        self.residual = residual
        norm_cls = _resolve_norm(norm) if norm is not None else None

        layers = []

        # Hidden layers
        current_in = in_features
        for _ in range(hidden_depth):
            layers.append(nn.Linear(current_in, hidden_features, dtype=FLOAT_DTYPE))
            if norm_cls is not None:
                layers.append(norm_cls(hidden_features, dtype=FLOAT_DTYPE))
            layers.append(activation())
            if dropout_p:
                layers.append(nn.Dropout(dropout_p))
            current_in = hidden_features

        # Output layer (no normalization, no activation, no dropout)
        layers.append(nn.Linear(current_in, out_features, dtype=FLOAT_DTYPE))

        self.net = nn.Sequential(*layers)

        if zero_init:
            self._init_zeros()

    def _init_zeros(self):
        """Set all Linear layer parameters (weight, bias) to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.residual:
            return x + self.net(x)
        else:
            return self.net(x)
