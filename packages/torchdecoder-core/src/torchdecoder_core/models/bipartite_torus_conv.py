"""Factored bipartite torus convolutions for BB-code Cascade backbones."""

import math

import torch
import torch.nn as nn

# Shape hints code:
# B = batch_size
# C = channels
# T = num_layers (time slices)
# ell, m = torus dimensions
# P = 2 data planes (left/right)


def _shift_time(x: torch.Tensor, dt: int) -> torch.Tensor:
    """
    Shift along the time axis (dim 2) with zero padding:
    ``out[:, :, t] = x[:, :, t + dt]`` (zero where ``t + dt`` is out of range).
    """
    if dt == 0:
        return x
    out = torch.zeros_like(x)
    if dt > 0:
        out[:, :, :-dt] = x[:, :, dt:]
    else:
        out[:, :, -dt:] = x[:, :, :dt]
    return out


class CheckToDataTorusConv(nn.Module):
    """
    Linear check -> data bipartite convolution step on the spacetime torus.

    Maps check-plane features shape=(B, in_channels, T, ell, m) to data-plane
    features shape=(B, out_channels, T, P, ell, m). The data qubit at torus
    site ``s`` of a plane gathers from its three check neighbors at sites
    ``s - offset`` (``offset`` ranging over the plane's check->data offset
    table) at temporal offsets {-1, 0}, each relation with an independent
    (out_channels, in_channels) weight matrix — translation-equivariant on
    the torus, zero-padded in time — plus a per-plane bias:
    6 spatial x 2 temporal = 12 relations.
    """

    time_taps = (-1, 0)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dataL_offsets: torch.Tensor,
        dataR_offsets: torch.Tensor,
    ):
        """
        Parameters
        ----------
            in_channels, out_channels : int
                Feature dimensions of the check (input) and data (output)
                nodes.

            dataL_offsets, dataR_offsets : torch.Tensor
                Torus offsets (drow, dcol) from a check site to its three
                left-/right-plane data neighbors, shape=(3, 2), int64.
        """
        super().__init__()
        offsets = torch.stack([dataL_offsets, dataR_offsets])  # (P, 3, 2)
        self.register_buffer("offsets", offsets, persistent=True)
        self._offsets = [[(int(dr), int(dc)) for dr, dc in plane] for plane in offsets]
        self.weight = nn.Parameter(
            torch.empty(2, 3, len(self.time_taps), out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.empty(2, out_channels))
        # Mirror nn.Conv3d's default init with fan_in = relations * in_channels:
        # each data node gathers from 3 spatial x 2 temporal relations.
        bound = 1.0 / math.sqrt(3 * len(self.time_taps) * in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        planes = []
        for p, plane_offsets in enumerate(self._offsets):
            contribs = []
            for j, (dr, dc) in enumerate(plane_offsets):
                # rolled[s] = x[s - offset]: the check neighbor of data site s.
                rolled = torch.roll(x, shifts=(dr, dc), dims=(3, 4))
                for k, dt in enumerate(self.time_taps):
                    contribs.append(
                        torch.einsum(
                            "oc,bcthw->bothw",
                            self.weight[p, j, k],
                            _shift_time(rolled, dt),
                        )
                    )
            planes.append(sum(contribs) + self.bias[p].view(1, -1, 1, 1, 1))
        return torch.stack(planes, dim=3)  # (B, C, T, P, ell, m)


class DataToCheckTorusConv(nn.Module):
    """
    Linear data -> check bipartite convolution step on the spacetime torus.

    Maps data-plane features shape=(B, in_channels, T, P, ell, m) to
    check-plane features shape=(B, out_channels, T, ell, m). The check at
    torus site ``s`` gathers from its three data neighbors at sites
    ``s + offset`` in each plane at temporal offsets {0, +1} (the complement
    of `CheckToDataTorusConv`'s taps, so the composition covers the symmetric
    {-1, 0, +1} window), each relation with an independent weight matrix:
    6 spatial x 2 temporal = 12 relations.
    """

    time_taps = (0, 1)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dataL_offsets: torch.Tensor,
        dataR_offsets: torch.Tensor,
    ):
        """
        Parameters
        ----------
            in_channels, out_channels : int
                Feature dimensions of the data (input) and check (output)
                nodes.

            dataL_offsets, dataR_offsets : torch.Tensor
                Torus offsets (drow, dcol) from a check site to its three
                left-/right-plane data neighbors, shape=(3, 2), int64.
        """
        super().__init__()
        offsets = torch.stack([dataL_offsets, dataR_offsets])  # (P, 3, 2)
        self.register_buffer("offsets", offsets, persistent=True)
        self._offsets = [[(int(dr), int(dc)) for dr, dc in plane] for plane in offsets]
        self.weight = nn.Parameter(
            torch.empty(2, 3, len(self.time_taps), out_channels, in_channels)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        # Each check gathers from 6 spatial x 2 temporal relations.
        bound = 1.0 / math.sqrt(6 * len(self.time_taps) * in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contribs = []
        for p, plane_offsets in enumerate(self._offsets):
            plane = x[:, :, :, p]  # (B, C, T, ell, m)
            for j, (dr, dc) in enumerate(plane_offsets):
                # rolled[s] = plane[s + offset]: the data neighbor of check s.
                rolled = torch.roll(plane, shifts=(-dr, -dc), dims=(3, 4))
                for k, dt in enumerate(self.time_taps):
                    contribs.append(
                        torch.einsum(
                            "oc,bcthw->bothw",
                            self.weight[p, j, k],
                            _shift_time(rolled, dt),
                        )
                    )
        return sum(contribs) + self.bias.view(1, -1, 1, 1, 1)  # (B, C, T, ell, m)


class BipartiteTorusConv(nn.Module):
    """
    Factored check -> check convolution on the spacetime torus
    (arXiv-2604.08358): two linear bipartite steps, check -> data -> check,
    each with 6 spatial x 2 temporal = 12 relations. The composition covers
    the dense check -> check neighborhood (22 spatial neighbors x 3 temporal
    offsets, plus self) with far fewer learned relations. Maps
    shape=(B, channels, T, ell, m) to the same shape.
    """

    def __init__(
        self,
        channels: int,
        dataL_offsets: torch.Tensor,
        dataR_offsets: torch.Tensor,
    ):
        """
        Parameters
        ----------
            channels : int
                Feature dimension of the check and data nodes.

            dataL_offsets, dataR_offsets : torch.Tensor
                Torus offsets (drow, dcol) from a check site to its three
                left-/right-plane data neighbors, shape=(3, 2), int64.
        """
        super().__init__()
        self.check_to_data = CheckToDataTorusConv(
            channels, channels, dataL_offsets, dataR_offsets
        )
        self.data_to_check = DataToCheckTorusConv(
            channels, channels, dataL_offsets, dataR_offsets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.data_to_check(self.check_to_data(x))
