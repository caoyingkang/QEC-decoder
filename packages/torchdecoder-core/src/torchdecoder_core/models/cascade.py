"""Cascade CNN logical decoder (arXiv-2604.08358)."""

import math

import torch
import torch.nn as nn

from ..utils.mlp import MLP
from .bipartite_torus_conv import BipartiteTorusConv, CheckToDataTorusConv
from .geometry import BBCodeGeometry, SurfaceCodeGeometry
from .logical_base import LogicalDecoderModel

# Shape hints code:
# B = batch_size
# H = hidden_dim
# T = num_layers (time slices)
# R, C = grid_height, grid_width (surface) or ell, m (BB torus)
# P = 2 data planes (BB only)
# O = num_obsers


class BottleneckBlock(nn.Module):
    """
    Pre-activation bottleneck residual block body: BN -> SiLU -> pointwise
    project down (H -> H/b), BN -> SiLU -> code-specific convolution in the
    bottleneck space, BN -> SiLU -> pointwise project up (H/b -> H). Returns
    the block output only; the caller applies the scaled residual connection.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int, conv: nn.Module):
        """
        Parameters
        ----------
            hidden_dim : int
                Width of the residual stream (H).

            bottleneck_dim : int
                Width of the bottleneck space the convolution operates in (H/b).

            conv : nn.Module
                Code-specific convolution mapping (B, bottleneck_dim, ...) to
                the same shape.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, bottleneck_dim, kernel_size=1),
            nn.BatchNorm3d(bottleneck_dim),
            nn.SiLU(),
            conv,
            nn.BatchNorm3d(bottleneck_dim),
            nn.SiLU(),
            nn.Conv3d(bottleneck_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Cascade(LogicalDecoderModel):
    """
    Cascade CNN logical decoder.

    The syndrome is scattered onto the detector spacetime grid, each detector
    bit is mapped to a `hidden_dim`-dim vector by a learned embedding (zero at
    grid sites hosting no detector), processed by `num_blocks` bottleneck
    residual blocks, scattered to data qubits by a final convolution,
    average-pooled over each logical observable's data-qubit support (and over
    time), and mapped to one logit per observable by a 2-layer MLP with hidden
    dimension `2 * hidden_dim`.

    The convolutions are the only code-specific component, selected by the
    geometry type. Surface code: standard 3D convolutions over
    (time, row, col) with zero padding. BB code: factored bipartite torus
    convolutions (check -> data -> check), with the final scatter convolution
    a single check -> data step whose output lives on the two data planes.

    All geometry tensors are registered as persistent buffers, so checkpoints
    are self-contained.
    """

    def __init__(
        self,
        geometry: SurfaceCodeGeometry | BBCodeGeometry,
        *,
        hidden_dim: int,
        num_blocks: int,
        bottleneck: int = 4,
    ):
        """
        Parameters
        ----------
            geometry : SurfaceCodeGeometry | BBCodeGeometry
                Spacetime-grid geometry of the circuit to decode; also selects
                the convolution backend.

            hidden_dim : int
                Width of the residual stream (`H` in the paper).

            num_blocks : int
                Number of bottleneck residual blocks (`L` in the paper).

            bottleneck : int
                Bottleneck factor `b`: convolutions operate in
                `hidden_dim // bottleneck` dimensions. Must divide hidden_dim.
        """
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be at least 1, but got {num_blocks}")
        if bottleneck < 1 or hidden_dim % bottleneck != 0:
            raise ValueError(
                f"bottleneck must be a positive divisor of hidden_dim, "
                f"but got hidden_dim={hidden_dim}, bottleneck={bottleneck}"
            )
        bottleneck_dim = hidden_dim // bottleneck
        match geometry:
            case BBCodeGeometry():
                height, width = geometry.ell, geometry.m
                offsets = (geometry.check_dataL_offsets, geometry.check_dataR_offsets)
                block_convs = [
                    BipartiteTorusConv(bottleneck_dim, *offsets)
                    for _ in range(num_blocks)
                ]
                scatter = CheckToDataTorusConv(hidden_dim, hidden_dim, *offsets)
            case SurfaceCodeGeometry():
                height, width = geometry.grid_height, geometry.grid_width
                block_convs = [
                    nn.Conv3d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1)
                    for _ in range(num_blocks)
                ]
                scatter = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            case _:
                raise TypeError(
                    f"Expected a SurfaceCodeGeometry or BBCodeGeometry, "
                    f"but got {type(geometry).__name__}"
                )
        super().__init__(geometry.num_detectors, geometry.num_observables)
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.grid_shape = (geometry.num_layers, height, width)
        num_layers = geometry.num_layers

        # Geometry buffers (persistent: checkpoints are self-contained).
        det_layers, det_rows, det_cols = geometry.detector_sites.unbind(dim=1)
        detector_flat_indices = (det_layers * height + det_rows) * width + det_cols
        self.register_buffer(
            "detector_flat_indices", detector_flat_indices, persistent=True
        )  # (num_chks,)

        spacetime_mask = torch.zeros(num_layers, height, width)
        spacetime_mask.view(-1)[detector_flat_indices] = 1.0
        self.register_buffer(
            "detector_spacetime_mask", spacetime_mask, persistent=True
        )  # (T, R, C)

        # Pooling weights: averaging over each observable's support sites and
        # over time is a weighted sum with these weights.
        observable_masks = geometry.observable_masks.float()  # (O, R, C) / (O, P, R, C)
        support_sizes = observable_masks.flatten(start_dim=1).sum(dim=1)
        pool_weights = observable_masks / (
            support_sizes.view(-1, *((1,) * (observable_masks.ndim - 1))) * num_layers
        )
        self.register_buffer(
            "observable_pool_weights", pool_weights, persistent=True
        )  # same shape as observable_masks

        self.embedding = nn.Embedding(2, hidden_dim)
        self.blocks = nn.ModuleList(
            BottleneckBlock(hidden_dim, bottleneck_dim, conv) for conv in block_convs
        )
        self.residual_scale = 1.0 / math.sqrt(2 * num_blocks)
        self.scatter_conv = nn.Sequential(
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(),
            scatter,
        )
        self.head = MLP(
            in_features=hidden_dim,
            out_features=1,
            hidden_features=2 * hidden_dim,
            hidden_depth=1,
            activation="SiLU",
            norm=None,
            dropout_p=None,
            residual=False,
        )

    def _embed(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Scatter syndromes onto the spacetime grid and embed each detector bit
        (zero where no detector). Returns shape=(B, H, T, R, C).
        """
        batch_size = syndromes.shape[0]
        num_layers, height, width = self.grid_shape
        bits = torch.zeros(
            batch_size,
            num_layers * height * width,
            dtype=torch.long,
            device=syndromes.device,
        )
        bits[:, self.detector_flat_indices] = syndromes.long()
        x = self.embedding(bits.view(batch_size, num_layers, height, width))
        return x.permute(0, 4, 1, 2, 3) * self.detector_spacetime_mask

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Residual blocks followed by the scatter-to-data-qubits convolution.
        Maps shape=(B, H, T, R, C) to the same shape (surface) or to
        shape=(B, H, T, P, R, C) (BB: one feature vector per data plane).
        """
        for block in self.blocks:
            x = x + self.residual_scale * block(x)
        return self.scatter_conv(x)

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        if syndromes.ndim != 2 or syndromes.shape[1] != self.num_chks:
            raise ValueError(
                f"Expected syndromes of shape (batch_size, {self.num_chks}), "
                f"but got shape {tuple(syndromes.shape)}"
            )
        x = self._backbone(self._embed(syndromes))  # (B, H, T, R, C) / (B, H, T, P, R, C)
        pooled = torch.einsum(
            "bhts,os->boh",
            x.flatten(start_dim=3),
            self.observable_pool_weights.flatten(start_dim=1),
        )  # (B, O, H)
        return self.head(pooled).squeeze(2)  # (B, O)
