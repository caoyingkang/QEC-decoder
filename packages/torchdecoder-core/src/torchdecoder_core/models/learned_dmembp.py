from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from ..utils.tensor_utils import (
    smooth_sign,
    smooth_min,
    leave_one_out_sign_product,
    leave_one_out_min,
    matmul_GF2,
)
from .base import DecoderModel, InferenceResult

EPS = 1e-6
BIG = 1e8

# Shape hints code:
# B = batch_size
# C = num_chks
# V = num_vars
# E = num_edges
# Δc = max_cn_deg
# Δv = max_vn_deg


class LearnedDMemBP(DecoderModel):
    """
    Disordered Memory BP decoder with trainable memory strength.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        num_iters: int,
        *,
        min_impl_method: Literal["smooth", "hard"],
        sign_impl_method: Literal["smooth", "hard"],
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

            min_impl_method : Literal["smooth", "hard"]
                Implementation method of the min function during training.
                Options: "smooth" (based on softmin) or "hard".
                Note that during inference, we always use "hard" min.

            sign_impl_method : Literal["smooth", "hard"]
                Implementation method of the sign function during training.
                Options: "smooth" (based on tanh), "hard".
                Note that during inference, we always use "hard" sign.
        """
        super().__init__(pcm, prior, num_iters)

        if min_impl_method not in ["smooth", "hard"]:
            raise ValueError(f"Invalid min_impl_method: {min_impl_method!r}")
        if sign_impl_method not in ["smooth", "hard"]:
            raise ValueError(f"Invalid sign_impl_method: {sign_impl_method!r}")
        self.min_impl_method = min_impl_method
        self.sign_impl_method = sign_impl_method

        # Build edge list from parity-check matrix.
        edge_to_cn, edge_to_vn = np.nonzero(pcm)
        num_edges = len(edge_to_cn)
        self.num_edges = num_edges

        # Build padded CN → edge table.
        # cn_edge_idx[i, k] = edge index for CN i's k-th neighbor (padded with num_edges).
        # cn_mask[i, k] = True if CN i has k-th neighbor, False otherwise.
        cn_deg = pcm.sum(axis=1).astype(int)
        self.max_cn_deg = int(cn_deg.max())
        cn_edge_idx = np.full(
            (self.num_chks, self.max_cn_deg), num_edges, dtype=np.int64
        )
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
        vn_edge_idx = np.full(
            (self.num_vars, self.max_vn_deg), num_edges, dtype=np.int64
        )
        vn_mask = np.zeros((self.num_vars, self.max_vn_deg), dtype=bool)
        vn_cursor = np.zeros(self.num_vars, dtype=int)
        for e in range(num_edges):
            j = edge_to_vn[e]
            k = vn_cursor[j]
            vn_edge_idx[j, k] = e
            vn_mask[j, k] = True
            vn_cursor[j] += 1

        # Register index buffers (derived from chkmat; not saved in checkpoints).
        # Since these are derived from chkmat, they are not part of the model's state_dict, and will not be saved in checkpoints.
        self.register_buffer(
            "edge_to_vn", torch.tensor(edge_to_vn, dtype=torch.long), persistent=False
        )  # (E,)
        self.register_buffer(
            "cn_edge_idx", torch.tensor(cn_edge_idx, dtype=torch.long), persistent=False
        )  # (C, Δc)
        self.register_buffer(
            "cn_mask", torch.tensor(cn_mask, dtype=torch.bool), persistent=False
        )  # (C, Δc)
        self.register_buffer(
            "vn_edge_idx", torch.tensor(vn_edge_idx, dtype=torch.long), persistent=False
        )  # (V, Δv)
        self.register_buffer(
            "vn_mask", torch.tensor(vn_mask, dtype=torch.bool), persistent=False
        )  # (V, Δv)
        if min_impl_method == "smooth" or sign_impl_method == "smooth":
            self.register_buffer(
                "cn_diag_mask",
                torch.eye(self.max_cn_deg, dtype=torch.bool),
                persistent=False,
            )  # (Δc, Δc)

        # Register prior LLRs.
        # We mark it as part of the model's state_dict, and will save it in checkpoints. In this way, one has the option to
        # use the prior_llr the model was trained with to benchmark its performance on other prior probabilities.
        prior = np.clip(prior, min=EPS, max=1 - EPS)
        prior_llr = np.log((1 - prior) / prior)
        self.register_buffer(
            "prior_llr", torch.tensor(prior_llr, dtype=torch.float32), persistent=True
        )  # (V,)

        # Initialize trainable parameter: memory strength.
        self.gamma = nn.Parameter(torch.zeros(self.num_vars))  # (V,)

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        device = syndromes.device
        batch_size = syndromes.shape[0]
        synd_sgn = (1 - 2 * syndromes).float()  # (B, C) ∈ {+1,-1}

        # Edge message arrays: (B, E + 1)
        # vn_to_cn[:, e], 0 <= e < num_edges, is the message along edge e from VN to CN.
        # cn_to_vn[:, e], 0 <= e < num_edges, is the message along edge e from CN to VN.
        # vn_to_cn[:, num_edges] is a dummy slot, so that vn_to_cn[:, self.cn_edge_idx] will not raise out-of-bounds error.
        # cn_to_vn[:, num_edges] is a dummy slot, so that cn_to_vn[:, self.vn_edge_idx] will not raise out-of-bounds error.
        vn_to_cn = torch.zeros(batch_size, self.num_edges + 1, device=device)
        cn_to_vn = torch.zeros(batch_size, self.num_edges + 1, device=device)

        # Initialize VN→CN messages with prior LLRs.
        vn_to_cn[:, : self.num_edges] = self.prior_llr[self.edge_to_vn]

        # Pre-expand flattened index tensors for scatter.
        cn_flat_idx = self.cn_edge_idx.reshape(1, -1).expand(
            batch_size, -1
        )  # (B, C * Δc)
        vn_flat_idx = self.vn_edge_idx.reshape(1, -1).expand(
            batch_size, -1
        )  # (B, V * Δv)

        cn_mask_3d = self.cn_mask.unsqueeze(0)  # (1, C, Δc)
        vn_mask_3d = self.vn_mask.unsqueeze(0)  # (1, V, Δv)

        llrs_list: list[torch.Tensor] = []
        prev_llrs = None

        for t in range(self.num_iters):
            # ==================== CN update ====================
            # Gather incoming messages at all CNs.
            msgs_cn = vn_to_cn[:, self.cn_edge_idx]  # (B, C, Δc)

            # Leave-one-out sign product.
            if (
                self.training and self.sign_impl_method == "smooth"
            ):  # 4D expansion + diagonal mask
                msgs_sgn = smooth_sign(msgs_cn).masked_fill(
                    ~cn_mask_3d, 1.0
                )  # (B, C, Δc)
                msgs_sgn_4d = msgs_sgn.unsqueeze(2).expand(
                    -1, -1, self.max_cn_deg, -1
                )  # (B, C, Δc, Δc)
                loo_sgn_prod = msgs_sgn_4d.masked_fill(self.cn_diag_mask, 1.0).prod(
                    dim=3
                )  # (B, C, Δc)
            else:  # Hard sign
                msgs_cn_masked = msgs_cn.masked_fill(~cn_mask_3d, 1.0)  # (B, C, Δc)
                loo_sgn_prod = leave_one_out_sign_product(
                    msgs_cn_masked, dim=2
                )  # (B, C, Δc)

            # Leave-one-out min abs.
            msgs_abs = msgs_cn.abs().masked_fill(~cn_mask_3d, BIG)  # (B, C, Δc)
            if (
                self.training and self.min_impl_method == "smooth"
            ):  # 4D expansion + diagonal mask
                msgs_abs_4d = msgs_abs.unsqueeze(2).expand(
                    -1, -1, self.max_cn_deg, -1
                )  # (B, C, Δc, Δc)
                loo_abs_min = smooth_min(
                    msgs_abs_4d.masked_fill(self.cn_diag_mask, BIG), dim=3
                )  # (B, C, Δc)
            else:  # Hard min
                loo_abs_min = leave_one_out_min(msgs_abs, dim=2)  # (B, C, Δc)

            # CN output messages.
            cn_out = synd_sgn.unsqueeze(2) * loo_sgn_prod * loo_abs_min  # (B, C, Δc)

            # Scatter CN outputs to edge array.
            # The values in cn_to_vn[:, num_edges] will be non-deterministic, but it's okay because they will be masked out during VN update.
            cn_to_vn.scatter_(1, cn_flat_idx, cn_out.reshape(batch_size, -1))

            # ==================== VN update ====================
            # Gather incoming messages at all VNs.
            msgs_vn = cn_to_vn[:, self.vn_edge_idx]  # (B, V, Δv)
            msgs_vn = msgs_vn.masked_fill(~vn_mask_3d, 0.0)  # (B, V, Δv)

            # Sum incoming messages per VN.
            incoming_sum = msgs_vn.sum(dim=2)  # (B, V)

            # Posterior LLR with memory.
            if t == 0:
                llrs = incoming_sum + self.prior_llr  # (B, V)
            else:
                llrs = (
                    incoming_sum
                    + (1.0 - self.gamma) * self.prior_llr
                    + self.gamma * prev_llrs
                )  # (B, V)

            llrs_list.append(llrs)
            prev_llrs = llrs

            if t < self.num_iters - 1:  # skip VN→CN messages in the last iteration
                # VN output messages.
                vn_out = llrs.unsqueeze(2) - msgs_vn  # (B, V, Δv)
                # Scatter VN outputs to edge array.
                # The values in vn_to_cn[:, num_edges] will be non-deterministic, but it's okay because they will be masked out during CN update.
                vn_to_cn.scatter_(1, vn_flat_idx, vn_out.reshape(batch_size, -1))

        return torch.stack(llrs_list, dim=0)  # (num_iters, B, V)

    def decode_inference(
        self, syndromes: torch.Tensor, chkmat: torch.Tensor
    ) -> InferenceResult:
        device = syndromes.device
        batch_size = syndromes.shape[0]
        synd_sgn = (1 - 2 * syndromes).float()  # (B, C) ∈ {+1,-1}

        vn_to_cn = torch.zeros(batch_size, self.num_edges + 1, device=device)
        cn_to_vn = torch.zeros(batch_size, self.num_edges + 1, device=device)

        vn_to_cn[:, : self.num_edges] = self.prior_llr[self.edge_to_vn]

        cn_flat_idx = self.cn_edge_idx.reshape(1, -1).expand(
            batch_size, -1
        )  # (B, C * Δc)
        vn_flat_idx = self.vn_edge_idx.reshape(1, -1).expand(
            batch_size, -1
        )  # (B, V * Δv)

        cn_mask_3d = self.cn_mask.unsqueeze(0)  # (1, C, Δc)
        vn_mask_3d = self.vn_mask.unsqueeze(0)  # (1, V, Δv)

        ehat = torch.zeros(batch_size, self.num_vars, device=device)
        converged_mask = torch.all(syndromes == 0, dim=1)  # (B,), bool
        decoding_iters = torch.zeros(batch_size, dtype=torch.long, device=device)
        prev_llrs = None

        for t in range(self.num_iters):
            msgs_cn = vn_to_cn[:, self.cn_edge_idx]  # (B, C, Δc)

            msgs_cn_masked = msgs_cn.masked_fill(~cn_mask_3d, 1.0)  # (B, C, Δc)
            loo_sgn_prod = leave_one_out_sign_product(
                msgs_cn_masked, dim=2
            )  # (B, C, Δc)

            msgs_abs = msgs_cn.abs().masked_fill(~cn_mask_3d, BIG)  # (B, C, Δc)
            loo_abs_min = leave_one_out_min(msgs_abs, dim=2)  # (B, C, Δc)

            cn_out = synd_sgn.unsqueeze(2) * loo_sgn_prod * loo_abs_min  # (B, C, Δc)
            cn_to_vn.scatter_(1, cn_flat_idx, cn_out.reshape(batch_size, -1))

            msgs_vn = cn_to_vn[:, self.vn_edge_idx]  # (B, V, Δv)
            msgs_vn = msgs_vn.masked_fill(~vn_mask_3d, 0.0)  # (B, V, Δv)

            incoming_sum = msgs_vn.sum(dim=2)  # (B, V)

            if t == 0:
                llrs = incoming_sum + self.prior_llr  # (B, V)
            else:
                llrs = (
                    incoming_sum
                    + (1.0 - self.gamma) * self.prior_llr
                    + self.gamma * prev_llrs
                )  # (B, V)
            prev_llrs = llrs

            hard_decisions = (llrs < 0).float()  # (B, V), float ∈ {0.0, 1.0}
            synd_pred = matmul_GF2(hard_decisions, chkmat.T)  # (B, C), int ∈ {0, 1}
            matched = torch.all(synd_pred == syndromes, dim=1)  # (B,)
            newly_converged = matched & ~converged_mask
            ehat[newly_converged] = hard_decisions[newly_converged]
            converged_mask |= matched
            decoding_iters[newly_converged] = t + 1
            if torch.all(converged_mask):
                break

            if t < self.num_iters - 1:
                vn_out = llrs.unsqueeze(2) - msgs_vn  # (B, V, Δv)
                vn_to_cn.scatter_(1, vn_flat_idx, vn_out.reshape(batch_size, -1))

        not_converged = ~converged_mask
        ehat[not_converged] = hard_decisions[not_converged]
        decoding_iters[not_converged] = self.num_iters

        return InferenceResult(ehat, converged_mask, decoding_iters)
