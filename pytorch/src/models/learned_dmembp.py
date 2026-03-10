from typing import Literal

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6
BIG = 1e8
FLOAT_DTYPE = torch.float32


class LearnedDMemBP(nn.Module):
    """
    Disordered Memory BP decoder with trainable memory strength.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        num_iters: int,
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
                Implementation method of the min function. Options: "smooth" (based on softmin), "hard" (using torch.amin).

            sign_impl_method : Literal["smooth", "hard"]
                Implementation method of the sign function. Options: "smooth" (based on tanh), "hard" (using torch.sign).
        """
        super().__init__()
        self.num_chks, self.num_vars = pcm.shape
        if num_iters < 1:
            raise ValueError(f"num_iters must be at least 1, but got {num_iters}")
        self.num_iters = num_iters

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
        self.register_buffer("edge_to_vn", torch.tensor(edge_to_vn, dtype=torch.long)) # (num_edges,)
        self.register_buffer("cn_edge_idx", torch.tensor(cn_edge_idx, dtype=torch.long)) # (num_chks, max_cn_deg)
        self.register_buffer("cn_mask", torch.tensor(cn_mask, dtype=torch.bool)) # (num_chks, max_cn_deg)
        self.register_buffer("vn_edge_idx", torch.tensor(vn_edge_idx, dtype=torch.long)) # (num_vars, max_vn_deg)
        self.register_buffer("vn_mask", torch.tensor(vn_mask, dtype=torch.bool)) # (num_vars, max_vn_deg)
        self.register_buffer("cn_diag_mask", torch.eye(self.max_cn_deg, dtype=torch.bool)) # (max_cn_deg, max_cn_deg)

        # Register prior LLRs.
        prior = np.clip(prior, min=EPS, max=1 - EPS)
        prior_llr = np.log((1 - prior) / prior)
        self.register_buffer("prior_llr", torch.tensor(prior_llr, dtype=FLOAT_DTYPE))  # (num_vars,)

        # Initialize trainable parameter: memory strength.
        self.gamma = nn.Parameter(torch.zeros(self.num_vars, dtype=FLOAT_DTYPE))  # (num_vars,)

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
        device = syndromes.device
        batch_size = syndromes.shape[0]
        synd_sgn = (1 - 2 * syndromes).to(FLOAT_DTYPE)  # (batch_size, num_chks) ∈ {+1,-1}

        # Edge message arrays: (batch_size, num_edges + 1)
        # vn_to_cn[:, e], 0 <= e < num_edges, is the message along edge e from VN to CN.
        # cn_to_vn[:, e], 0 <= e < num_edges, is the message along edge e from CN to VN.
        # vn_to_cn[:, num_edges] is a dummy slot, so that vn_to_cn[:, self.cn_edge_idx] will not raise out-of-bounds error.
        # cn_to_vn[:, num_edges] is a dummy slot, so that cn_to_vn[:, self.vn_edge_idx] will not raise out-of-bounds error.
        vn_to_cn = torch.zeros(batch_size, self.num_edges + 1, device=device, dtype=FLOAT_DTYPE)
        cn_to_vn = torch.zeros(batch_size, self.num_edges + 1, device=device, dtype=FLOAT_DTYPE)

        # Initialize VN→CN messages with prior LLRs.
        vn_to_cn[:, :self.num_edges] = self.prior_llr[self.edge_to_vn]

        # Pre-expand flattened index tensors for scatter.
        cn_flat_idx = self.cn_edge_idx.reshape(1, -1).expand(batch_size, -1)  # (batch_size, num_chks * max_cn_deg)
        vn_flat_idx = self.vn_edge_idx.reshape(1, -1).expand(batch_size, -1)  # (batch_size, num_vars * max_vn_deg)

        cn_mask_3d = self.cn_mask.unsqueeze(0)  # (1, num_chks, max_cn_deg)
        vn_mask_3d = self.vn_mask.unsqueeze(0)  # (1, num_vars, max_vn_deg)

        llrs_list: list[torch.Tensor] = []
        prev_llrs = None

        for t in range(self.num_iters):
            # ==================== CN update ====================
            # Gather incoming messages at all CNs.
            msgs_cn = vn_to_cn[:, self.cn_edge_idx]  # (batch_size, num_chks, max_cn_deg)
            msgs_sgn = self.sign_func(msgs_cn).masked_fill(~cn_mask_3d, 1.0)  # (batch_size, num_chks, max_cn_deg)
            msgs_abs = msgs_cn.abs().masked_fill(~cn_mask_3d, BIG)  # (batch_size, num_chks, max_cn_deg)

            # Leave-one-out sign product via 4D expansion + diagonal mask.
            msgs_sgn_4d = msgs_sgn.unsqueeze(2).expand(-1, -1, self.max_cn_deg, -1)  # (batch_size, num_chks, max_cn_deg, max_cn_deg)
            loo_sgn_prod = msgs_sgn_4d.masked_fill(self.cn_diag_mask, 1.0).prod(dim=3)  # (batch_size, num_chks, max_cn_deg)

            # Leave-one-out min abs via 4D expansion + diagonal mask.
            msgs_abs_4d = msgs_abs.unsqueeze(2).expand(-1, -1, self.max_cn_deg, -1)  # (batch_size, num_chks, max_cn_deg, max_cn_deg)
            loo_abs_min = self.min_func(msgs_abs_4d.masked_fill(self.cn_diag_mask, BIG), dim=3)  # (batch_size, num_chks, max_cn_deg)

            # CN output messages.
            cn_out = synd_sgn.unsqueeze(2) * loo_sgn_prod * loo_abs_min   # (batch_size, num_chks, max_cn_deg)

            # Scatter CN outputs to edge array.
            # The values in cn_to_vn[:, num_edges] will be non-deterministic, but it's okay because they will be masked out during VN update.
            cn_to_vn.scatter_(1, cn_flat_idx, cn_out.reshape(batch_size, -1))

            # ==================== VN update ====================
            # Gather incoming messages at all VNs.
            msgs_vn = cn_to_vn[:, self.vn_edge_idx]  # (batch_size, num_vars, max_vn_deg)
            msgs_vn = msgs_vn.masked_fill(~vn_mask_3d, 0.0)  # (batch_size, num_vars, max_vn_deg)

            # Sum incoming messages per VN.
            incoming_sum = msgs_vn.sum(dim=2)  # (batch_size, num_vars)

            # Posterior LLR with memory.
            if t == 0:
                llrs = incoming_sum + self.prior_llr  # (batch_size, num_vars)
            else:
                llrs = incoming_sum + \
                    (1.0 - self.gamma) * self.prior_llr + \
                    self.gamma * prev_llrs  # (batch_size, num_vars)

            llrs_list.append(llrs)
            prev_llrs = llrs

            if t < self.num_iters - 1:  # skip VN→CN messages in the last iteration
                # VN output messages.
                vn_out = llrs.unsqueeze(2) - msgs_vn  # (batch_size, num_vars, max_vn_deg)
                # Scatter VN outputs to edge array.
                # The values in vn_to_cn[:, num_edges] will be non-deterministic, but it's okay because they will be masked out during CN update.
                vn_to_cn.scatter_(1, vn_flat_idx, vn_out.reshape(batch_size, -1))

        return torch.stack(llrs_list, dim=0)  # (num_iters, batch_size, num_vars)
