"""Torch adapter for the spacetime-grid geometry of rotated-surface-code circuits."""

import torch

from qecdec.circuits import RotatedSurfaceCodeBase


class SurfaceCodeGeometry:
    """
    Thin torch adapter around the grid geometry that a rotated-surface-code
    circuit exposes (``grid_shape``, ``detector_grid_sites``,
    ``detector_site_mask``, ``observable_grid_masks``): converts the numpy
    geometry to torch tensors and provides the syndrome <-> grid scatter/gather.

    Attributes
    ----------
        num_detectors : int
            Number of detectors in the circuit.
        num_observables : int
            Number of logical observables in the circuit.
        num_layers : int
            Number of detector time slices (measurement rounds + 1
            final-measurement slice).
        grid_height : int
            Number of distinct detector y coordinates.
        grid_width : int
            Number of distinct detector x coordinates.
        detector_sites : torch.Tensor
            Grid site of each detector, shape=(num_detectors, 3), int64.
            Each row is (layer, row, col).
        detector_site_mask : torch.Tensor
            Spatial sites hosting a detector (in any layer),
            shape=(grid_height, grid_width), bool.
        observable_masks : torch.Tensor
            Per-observable data-qubit support masks,
            shape=(num_observables, grid_height, grid_width), bool.
    """

    def __init__(self, circuit: RotatedSurfaceCodeBase) -> None:
        """
        Parameters
        ----------
            circuit : RotatedSurfaceCodeBase
                A rotated-surface-code circuit exposing grid geometry.
        """
        if not isinstance(circuit, RotatedSurfaceCodeBase):
            raise TypeError(
                "Expected a RotatedSurfaceCodeBase circuit exposing grid geometry, "
                f"but got {type(circuit).__name__}"
            )

        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        self.num_layers, self.grid_height, self.grid_width = circuit.grid_shape

        # Copy so the tensors don't alias the circuit's cached numpy arrays.
        self.detector_sites = torch.from_numpy(circuit.detector_grid_sites.copy())
        self.detector_site_mask = torch.from_numpy(circuit.detector_site_mask.copy())
        self.observable_masks = torch.from_numpy(circuit.observable_grid_masks.copy())

        det_layers, det_rows, det_cols = self.detector_sites.unbind(dim=1)
        self._flat_indices = (
            det_layers * self.grid_height + det_rows
        ) * self.grid_width + det_cols

    def syndrome_to_grid(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Scatter flat syndromes onto the spacetime grid (zero where no detector).

        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_detectors).

        Returns
        -------
            grid : torch.Tensor
                Same dtype as the input,
                shape=(batch_size, num_layers, grid_height, grid_width).
        """
        if syndromes.ndim != 2 or syndromes.shape[1] != self.num_detectors:
            raise ValueError(
                f"Expected syndromes of shape (batch_size, {self.num_detectors}), "
                f"but got shape {tuple(syndromes.shape)}"
            )
        batch_size = syndromes.shape[0]
        grid = syndromes.new_zeros(
            (batch_size, self.num_layers * self.grid_height * self.grid_width)
        )
        grid[:, self._flat_indices.to(syndromes.device)] = syndromes
        return grid.view(
            batch_size, self.num_layers, self.grid_height, self.grid_width
        )

    def grid_to_syndrome(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Gather flat syndromes from the spacetime grid (inverse of
        ``syndrome_to_grid``).

        Parameters
        ----------
            grid : torch.Tensor
                shape=(batch_size, num_layers, grid_height, grid_width).

        Returns
        -------
            syndromes : torch.Tensor
                Same dtype as the input, shape=(batch_size, num_detectors).
        """
        expected = (self.num_layers, self.grid_height, self.grid_width)
        if grid.ndim != 4 or tuple(grid.shape[1:]) != expected:
            raise ValueError(
                f"Expected grid of shape (batch_size, {expected[0]}, {expected[1]}, "
                f"{expected[2]}), but got shape {tuple(grid.shape)}"
            )
        return grid.reshape(grid.shape[0], -1)[:, self._flat_indices.to(grid.device)]
