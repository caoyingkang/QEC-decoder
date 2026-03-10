"""
Multi-dimensional Disordered Memory BP decoder.

This module extends LearnedDMemBP by replacing scalar messages with vector-valued messages
of dimension `msg_features`. Each component of the vector undergoes the same DMemBP updates 
as in LearnedDMemBP (min-sum update at check nodes, disordered memory at variable nodes), but 
every message passes through an MLP before being sent. After each iteration, the LLR values
are calculated per message component, as if each component makes independent predictions; those 
LLR components are then averaged to produce the final scalar LLR per variable node, as if a 
majority vote is performed.

Design choices and rationale:
- There are two separate MLPs, one for each message direction (VN-to-CN and CN-to-VN). All messages
  in the same direction share the same MLP to reduce the number of trainable parameters.
- The MLPs are augmented with residual connections and are initialized to all-zero weights and biases,
  so that the MLPs start as identity functions. In this case, each message component is completely 
  independent of the other components, and the whole network reduces to LearnedDMemBP. As training 
  proceeds, the MLPs can learn to mix components, allowing information flow between the parallel 
  DMemBP instances.
- Memory strength (gamma) can be shared across message components (gamma_shared=True) or vary per
  component (gamma_shared=False).
- Memory strength can be initialized to a single value (if gamma_init is a float) or sampled uniformly 
  from an interval (if gamma_init is a list of two floats).

Notes:
- Setting `mlp_norm="LayerNorm"` will subtract the mean across the message feature dimension, which can 
  remove important sign information: It is possible that for some (CN, VN) pairs all message components 
  are positive (they all agree that the VN is likely error-free), while for some other (CN, VN) pairs 
  all message components are negative (they all agree that the VN is likely erroneous). LayerNorm removes 
  this mean and can potentially degrade decoding performance. Consider setting `mlp_norm="RMSNorm"` 
  (preserves sign) or `mlp_norm=None` (no normalization) instead.
"""
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ..utils.mlp import MLP

EPS = 1e-6
BIG = 1e8
FLOAT_DTYPE = torch.float32


class MultiDMemBP(nn.Module):
    """
    Multi-dimensional Disordered Memory BP decoder.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        num_iters: int,
        msg_features: int,
        mlp_hidden_features: int,
        mlp_hidden_depth: int,
        mlp_activation: nn.Module,
        mlp_norm: Optional[str],
        mlp_dropout_p: Optional[float],
        min_impl_method: Literal["smooth", "hard"],
        sign_impl_method: Literal["smooth", "hard"],
        gamma_shared: bool,
        gamma_init: Union[float, list[float]],
    ):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity-check matrix, shape=(num_chks, num_vars), integer ∈ {0,1} or bool

            prior : ndarray
                Prior probabilities of errors, shape=(num_vars,), float

            num_iters : int
                Number of BP iterations.

            msg_features : int
                Number of features in (i.e., dimension of) the message vector.

            mlp_hidden_features : int
                Number of features in each hidden layer of the MLP.

            mlp_hidden_depth : int
                Number of hidden layers of the MLP. Must be at least 1.

            mlp_activation : nn.Module
                Activation function to use in the hidden layers of the MLP.

            mlp_norm : str | None
                Normalization to use in the hidden layers of the MLP. If None, no normalization.
                Supported options are "LayerNorm" and "RMSNorm".

            mlp_dropout_p : Optional[float]
                Dropout probability for the hidden layers of the MLP. If None, no dropout is applied.

            min_impl_method : Literal["smooth", "hard"]
                Implementation method of the min function. Options: "smooth" (based on softmin), "hard" (using torch.amin).

            sign_impl_method : Literal["smooth", "hard"]
                Implementation method of the sign function. Options: "smooth" (based on tanh), "hard" (using torch.sign).

            gamma_shared : bool
                If True, memory strengths vary per variable node but are shared across message components.
                If False, memory strengths vary per variable node and per message component.

            gamma_init : float | list[float]
                Initial value(s) for gamma. If a single float, all gamma values are set to it.
                If a list of two floats [a, b], each gamma is sampled independently and uniformly from [a, b].
        """
        super().__init__()
        self.num_chks, self.num_vars = pcm.shape
        if num_iters < 1:
            raise ValueError(f"num_iters must be at least 1, but got {num_iters}")
        if msg_features < 1:
            raise ValueError(f"msg_features must be at least 1, but got {msg_features}")
        self.num_iters = num_iters
        self.msg_features = msg_features

        self.v2c_mlp = MLP(
            in_features=msg_features,
            out_features=msg_features,
            hidden_features=mlp_hidden_features,
            hidden_depth=mlp_hidden_depth,
            activation=mlp_activation,
            norm=mlp_norm,
            dropout_p=mlp_dropout_p,
            zero_init=True,
            residual=True,
        )
        self.c2v_mlp = MLP(
            in_features=msg_features,
            out_features=msg_features,
            hidden_features=mlp_hidden_features,
            hidden_depth=mlp_hidden_depth,
            activation=mlp_activation,
            norm=mlp_norm,
            dropout_p=mlp_dropout_p,
            zero_init=True,
            residual=True,
        )

        if min_impl_method == "smooth":
            from ..utils.tensor_utils import smooth_min
            self.min_func = smooth_min
        elif min_impl_method == "hard":
            self.min_func = torch.amin
        else:
            raise ValueError(f"Invalid min_impl_method: {min_impl_method!r}")

        if sign_impl_method == "smooth":
            from ..utils.tensor_utils import smooth_sign
            self.sign_func = smooth_sign
        elif sign_impl_method == "hard":
            self.sign_func = torch.sign
        else:
            raise ValueError(f"Invalid sign_impl_method: {sign_impl_method!r}")

        # Build edge list from parity-check matrix.
        edge_to_cn, edge_to_vn = np.nonzero(pcm)
        num_edges = len(edge_to_cn)
        self.num_edges = num_edges

        # Build padded CN → edge table.
        # cn_edge_idx[i, k] = edge index for CN i's k-th neighbor (padded with num_edges).
        # cn_mask[i, k] = True if CN i has k-th neighbor, False otherwise.
        cn_deg = pcm.sum(axis=1).astype(int)
        self.max_cn_deg = int(cn_deg.max())
        cn_edge_idx = np.full((self.num_chks, self.max_cn_deg), num_edges, dtype=np.int64)
        cn_mask = np.zeros((self.num_chks, self.max_cn_deg), dtype=bool)
        cn_cursor = np.zeros(self.num_chks, dtype=int)
        for e in range(num_edges):
            i = edge_to_cn[e]
            k = cn_cursor[i]
            cn_edge_idx[i, k] = e
            cn_mask[i, k] = True
            cn_cursor[i] += 1

        # Build padded VN → edge table.
        # vn_edge_idx[j, k] = edge index for VN j's k-th neighbor (padded with num_edges).
        # vn_mask[j, k] = True if VN j has k-th neighbor, False otherwise.
        vn_deg = pcm.sum(axis=0).astype(int)
        self.max_vn_deg = int(vn_deg.max())
        vn_edge_idx = np.full((self.num_vars, self.max_vn_deg), num_edges, dtype=np.int64)
        vn_mask = np.zeros((self.num_vars, self.max_vn_deg), dtype=bool)
        vn_cursor = np.zeros(self.num_vars, dtype=int)
        for e in range(num_edges):
            j = edge_to_vn[e]
            k = vn_cursor[j]
            vn_edge_idx[j, k] = e
            vn_mask[j, k] = True
            vn_cursor[j] += 1

        # Register index buffers.
        self.register_buffer("edge_to_vn", torch.tensor(edge_to_vn, dtype=torch.long))  # (num_edges,)
        self.register_buffer("cn_edge_idx", torch.tensor(cn_edge_idx, dtype=torch.long))  # (num_chks, max_cn_deg)
        self.register_buffer("cn_mask", torch.tensor(cn_mask, dtype=torch.bool))  # (num_chks, max_cn_deg)
        self.register_buffer("vn_edge_idx", torch.tensor(vn_edge_idx, dtype=torch.long))  # (num_vars, max_vn_deg)
        self.register_buffer("vn_mask", torch.tensor(vn_mask, dtype=torch.bool))  # (num_vars, max_vn_deg)
        self.register_buffer("cn_diag_mask", torch.eye(self.max_cn_deg, dtype=torch.bool))  # (max_cn_deg, max_cn_deg)

        # Register prior LLRs.
        prior = np.clip(prior, min=EPS, max=1 - EPS)
        prior_llr = np.log((1 - prior) / prior)
        self.register_buffer("prior_llr", torch.tensor(prior_llr, dtype=FLOAT_DTYPE))  # (num_vars,)

        # Initialize trainable parameter: memory strength.
        self.gamma_shared = gamma_shared
        self.gamma = nn.Parameter(self._init_gamma(gamma_init))  # (num_vars,) or (num_vars, msg_features)

    def _init_gamma(self, gamma_init: Union[float, list[float]]) -> torch.Tensor:
        if self.gamma_shared:
            shape = (self.num_vars,)
        else:
            shape = (self.num_vars, self.msg_features)

        # Support int or float. Note that bool is a subclass of int, but we don't want to treat it as such.
        if isinstance(gamma_init, (int, float)) and not isinstance(gamma_init, bool):
            return torch.full(shape, float(gamma_init), dtype=FLOAT_DTYPE)
        # Support any class with __getitem__ and length 2.
        elif hasattr(gamma_init, "__getitem__") and hasattr(gamma_init, "__len__") and len(gamma_init) == 2:
            low, high = float(gamma_init[0]), float(gamma_init[1])
            if low > high:
                raise ValueError(f"Invalid interval, got {gamma_init!r}")
            return torch.empty(shape, dtype=FLOAT_DTYPE).uniform_(low, high)
        else:
            raise ValueError(f"gamma_init must be a float or a list of two floats [low, high], got {gamma_init!r}")

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_chks), int ∈ {0,1}

        Returns
        -------
            llrs : torch.Tensor
                LLR values at all BP iterations, shape=(num_iters, batch_size, num_vars), float
        """
        # Shape hints code:
        # B = batch_size
        # C = num_chks
        # V = num_vars
        # E = num_edges
        # M = msg_features
        # Δc = max_cn_deg
        # Δv = max_vn_deg

        device = syndromes.device
        batch_size = syndromes.shape[0]
        synd_sgn = (1 - 2 * syndromes).to(FLOAT_DTYPE)  # (B, C)
        mf = self.msg_features

        # Edge message arrays: (B, E + 1, M)
        vn_to_cn = torch.zeros(batch_size, self.num_edges + 1, mf, device=device, dtype=FLOAT_DTYPE)
        cn_to_vn = torch.zeros(batch_size, self.num_edges + 1, mf, device=device, dtype=FLOAT_DTYPE)

        # Initialize VN→CN messages with prior LLRs (broadcast to all components).
        vn_to_cn[:, :self.num_edges, :] = self.prior_llr[self.edge_to_vn].unsqueeze(0).unsqueeze(-1)

        # Pre-expand flattened index tensors for scatter.
        cn_flat_idx = self.cn_edge_idx.reshape(1, -1, 1).expand(batch_size, -1, mf)  # (B, C * Δc, M)
        vn_flat_idx = self.vn_edge_idx.reshape(1, -1, 1).expand(batch_size, -1, mf)  # (B, V * Δv, M)

        cn_mask_4d = self.cn_mask.unsqueeze(0).unsqueeze(-1)  # (1, C, Δc, 1)
        vn_mask_4d = self.vn_mask.unsqueeze(0).unsqueeze(-1)  # (1, V, Δv, 1)

        llrs_list: list[torch.Tensor] = []  # List of (B, V) tensors, one for each BP iteration.
        prev_llrs_all_components = None  # Will hold tensor of shape (B, V, M)

        for t in range(self.num_iters):
            # ==================== CN update ====================
            # Gather incoming messages at all CNs.
            msgs_cn = vn_to_cn[:, self.cn_edge_idx, :]  # (B, C, Δc, M)
            msgs_sgn = self.sign_func(msgs_cn).masked_fill(~cn_mask_4d, 1.0)  # (B, C, Δc, M)
            msgs_abs = msgs_cn.abs().masked_fill(~cn_mask_4d, BIG)  # (B, C, Δc, M)

            # Leave-one-out sign product via 5D expansion + diagonal mask.
            msgs_sgn_5d = msgs_sgn.unsqueeze(2).expand(-1, -1, self.max_cn_deg, -1, -1)  # (B, C, Δc, Δc, M)
            cn_diag_mask_5d = self.cn_diag_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, Δc, Δc, 1)
            loo_sgn_prod = msgs_sgn_5d.masked_fill(cn_diag_mask_5d, 1.0).prod(dim=3)  # (B, C, Δc, M)

            # Leave-one-out min abs via 5D expansion + diagonal mask.
            msgs_abs_5d = msgs_abs.unsqueeze(2).expand(-1, -1, self.max_cn_deg, -1, -1)  # (B, C, Δc, Δc, M)
            loo_abs_min = self.min_func(msgs_abs_5d.masked_fill(cn_diag_mask_5d, BIG), dim=3)  # (B, C, Δc, M)

            # CN output messages.
            cn_out = synd_sgn.unsqueeze(-1).unsqueeze(-1) * loo_sgn_prod * loo_abs_min  # (B, C, Δc, M)

            # Apply MLP before sending CN→VN.
            cn_out_flat = cn_out.reshape(-1, mf)  # (B * C * Δc, M)
            cn_out_flat: torch.Tensor = self.c2v_mlp(cn_out_flat)  # (B * C * Δc, M)
            cn_out = cn_out_flat.reshape(batch_size, self.num_chks, self.max_cn_deg, mf)  # (B, C, Δc, M)

            # Scatter CN outputs to edge array.
            cn_out_for_scatter = cn_out.reshape(batch_size, -1, mf)  # (B, C * Δc, M)
            cn_to_vn.scatter_(1, cn_flat_idx, cn_out_for_scatter)

            # ==================== VN update ====================
            # Gather incoming messages at all VNs.
            msgs_vn = cn_to_vn[:, self.vn_edge_idx, :]  # (B, V, Δv, M)
            msgs_vn = msgs_vn.masked_fill(~vn_mask_4d, 0.0)  # (B, V, Δv, M)

            # Sum incoming messages per VN.
            incoming_sum = msgs_vn.sum(dim=2)  # (B, V, M)

            # Posterior LLR with memory.
            pllr = self.prior_llr.unsqueeze(0).unsqueeze(-1)  # (1, V, 1)
            if self.gamma_shared:
                g = self.gamma.unsqueeze(0).unsqueeze(-1)  # (1, V, 1)
            else:
                g = self.gamma.unsqueeze(0)  # (1, V, M)
            if t == 0:
                llrs_all_components = incoming_sum + pllr  # (B, V, M)
            else:
                llrs_all_components = incoming_sum + (1.0 - g) * pllr + g * prev_llrs_all_components  # (B, V, M)
            prev_llrs_all_components = llrs_all_components

            # Soft majority vote: average over LLRs along the message feature dimension.
            llrs = llrs_all_components.mean(dim=2)  # (B, V)
            llrs_list.append(llrs)

            if t < self.num_iters - 1:  # skip VN→CN messages in the last iteration
                # VN output messages.
                vn_out = llrs_all_components.unsqueeze(2) - msgs_vn  # (B, V, Δv, M)

                # Apply MLP before sending VN→CN.
                vn_out_flat = vn_out.reshape(-1, mf)  # (B * V * Δv, M)
                vn_out_flat: torch.Tensor = self.v2c_mlp(vn_out_flat)  # (B * V * Δv, M)
                vn_out = vn_out_flat.reshape(batch_size, self.num_vars, self.max_vn_deg, mf)  # (B, V, Δv, M)

                # Scatter VN outputs to edge array.
                vn_out_for_scatter = vn_out.reshape(batch_size, -1, mf)  # (B, V * Δv, M)
                vn_to_cn.scatter_(1, vn_flat_idx, vn_out_for_scatter)

        return torch.stack(llrs_list, dim=0)  # (num_iters, B, V)
