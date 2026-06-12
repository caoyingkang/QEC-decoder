"""Torch adapter for the torus geometry of bivariate-bicycle-code circuits."""

import torch

from qecdec.circuits import BBCode_Circuit


class BBCodeGeometry:
    """
    Thin torch adapter around the torus geometry that a BB-code circuit
    exposes (``grid_shape``, ``detector_grid_sites``, ``observable_grid_masks``,
    check<->data offset tables): converts the numpy geometry to torch tensors
    and provides the syndrome <-> grid scatter/gather. In-basis detectors fill
    the ``(ell, m)`` torus densely in every layer, so the scatter is a plain
    reshape.

    Attributes
    ----------
        num_detectors : int
            Number of detectors in the circuit.
        num_observables : int
            Number of logical observables in the circuit.
        num_layers : int
            Number of detector time slices (measurement cycles + 1
            final-measurement slice).
        ell : int
            Number of torus rows.
        m : int
            Number of torus columns.
        detector_sites : torch.Tensor
            Grid site of each detector, shape=(num_detectors, 3), int64.
            Each row is (layer, row, col).
        observable_masks : torch.Tensor
            Per-observable data-qubit support masks,
            shape=(num_observables, 2, ell, m), bool. The second axis indexes
            the left/right data plane.
        check_dataL_offsets : torch.Tensor
            Torus offsets (drow, dcol) from an in-basis check site to its
            three left-plane data neighbors, shape=(3, 2), int64.
        check_dataR_offsets : torch.Tensor
            Same for the right-plane data neighbors, shape=(3, 2), int64.
    """

    def __init__(self, circuit: BBCode_Circuit) -> None:
        """
        Parameters
        ----------
            circuit : BBCode_Circuit
                A BB-code circuit exposing torus geometry. Must have been
                built with ``filter_detectors=True`` (in-basis detectors only).
        """
        if not isinstance(circuit, BBCode_Circuit):
            raise TypeError(
                "Expected a BBCode_Circuit exposing torus geometry, "
                f"but got {type(circuit).__name__}"
            )
        if not circuit.filter_detectors:
            raise ValueError(
                "BBCodeGeometry requires an in-basis circuit "
                "(filter_detectors=True)"
            )

        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        self.num_layers, self.ell, self.m = circuit.grid_shape

        self.detector_sites = torch.from_numpy(circuit.detector_grid_sites)
        self.observable_masks = torch.from_numpy(circuit.observable_grid_masks)
        self.check_dataL_offsets = torch.from_numpy(circuit.check_dataL_offsets)
        self.check_dataR_offsets = torch.from_numpy(circuit.check_dataR_offsets)

    def syndrome_to_grid(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Arrange flat syndromes onto the spacetime torus grid.

        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_detectors).

        Returns
        -------
            grid : torch.Tensor
                Same dtype as the input,
                shape=(batch_size, num_layers, ell, m).
        """
        if syndromes.ndim != 2 or syndromes.shape[1] != self.num_detectors:
            raise ValueError(
                f"Expected syndromes of shape (batch_size, {self.num_detectors}), "
                f"but got shape {tuple(syndromes.shape)}"
            )
        return syndromes.view(-1, self.num_layers, self.ell, self.m)

    def grid_to_syndrome(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Gather flat syndromes from the spacetime torus grid (inverse of
        ``syndrome_to_grid``).

        Parameters
        ----------
            grid : torch.Tensor
                shape=(batch_size, num_layers, ell, m).

        Returns
        -------
            syndromes : torch.Tensor
                Same dtype as the input, shape=(batch_size, num_detectors).
        """
        expected = (self.num_layers, self.ell, self.m)
        if grid.ndim != 4 or tuple(grid.shape[1:]) != expected:
            raise ValueError(
                f"Expected grid of shape (batch_size, {expected[0]}, {expected[1]}, "
                f"{expected[2]}), but got shape {tuple(grid.shape)}"
            )
        return grid.reshape(grid.shape[0], -1)
