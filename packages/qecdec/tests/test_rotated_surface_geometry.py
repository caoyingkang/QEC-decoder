from functools import cache

import numpy as np
import pytest

from qecdec.circuits import CIRCUITS_REGISTRY, RotatedSurfaceCodeBase

CIRCUIT_CASES = [
    (name, d, basis)
    for name in ["RotatedSurfaceCode_Phenom", "RotatedSurfaceCode_Circuit"]
    for d in [3, 5]
    for basis in ["Z", "X"]
]


@cache
def make_circuit(circuit_name: str, d: int, basis: str) -> RotatedSurfaceCodeBase:
    return CIRCUITS_REGISTRY[circuit_name].with_uniform_error_rate(
        0.01, d=d, rounds=d, basis=basis
    )


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_grid_shape(circuit_name: str, d: int, basis: str) -> None:
    circuit = make_circuit(circuit_name, d, basis)
    num_layers, height, width = circuit.grid_shape
    # rounds of stabilizer measurement + the final data-qubit measurement
    assert num_layers == d + 1
    # checks of one basis span d+1 columns and d-1 rows (or transposed)
    assert {height, width} == {d - 1, d + 1}
    assert num_layers * height * width >= circuit.num_detectors


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_detector_grid_sites_match_dem_coords(
    circuit_name: str, d: int, basis: str
) -> None:
    """Ranking the DEM detector coordinates (x -> col, y -> row, t -> round) must
    reproduce the construction-derived grid sites exactly. This guards the
    assumed detector ordering against stim ground truth."""
    circuit = make_circuit(circuit_name, d, basis)
    coords = circuit.detector_coords
    assert coords.shape == (circuit.num_detectors, 3)
    xs, ys, ts = coords.T
    expected = np.stack(
        [
            np.searchsorted(np.unique(ts), ts),
            np.searchsorted(np.unique(ys), ys),
            np.searchsorted(np.unique(xs), xs),
        ],
        axis=1,
    )
    assert np.array_equal(circuit.detector_grid_sites, expected)


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_detector_grid_sites_unique(circuit_name: str, d: int, basis: str) -> None:
    circuit = make_circuit(circuit_name, d, basis)
    sites = circuit.detector_grid_sites
    assert sites.shape == (circuit.num_detectors, 3)
    num_layers, height, width = circuit.grid_shape
    assert np.all(sites >= 0)
    assert np.all(sites < [num_layers, height, width])
    flat = (sites[:, 0] * height + sites[:, 1]) * width + sites[:, 2]
    assert len(np.unique(flat)) == circuit.num_detectors


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_detector_site_mask(circuit_name: str, d: int, basis: str) -> None:
    circuit = make_circuit(circuit_name, d, basis)
    _, height, width = circuit.grid_shape
    mask = circuit.detector_site_mask
    assert mask.shape == (height, width)
    assert mask.sum() == len(circuit.basis_check_coos)
    assert np.all(mask[circuit.detector_grid_sites[:, 1], circuit.detector_grid_sites[:, 2]])


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_observable_grid_masks(circuit_name: str, d: int, basis: str) -> None:
    circuit = make_circuit(circuit_name, d, basis)
    _, height, width = circuit.grid_shape
    masks = circuit.observable_grid_masks
    assert masks.shape == (circuit.num_observables, height, width)
    assert masks.dtype == np.bool_
    for i in range(circuit.num_observables):
        assert masks[i].any(), f"observable {i} mask is empty"
        assert not (masks[i] & circuit.detector_site_mask).any(), (
            f"observable {i} mask overlaps detector sites"
        )


@pytest.mark.parametrize("circuit_name", ["RotatedSurfaceCode_Phenom", "RotatedSurfaceCode_Circuit"])
def test_observable_mask_hugs_support_boundary(circuit_name: str) -> None:
    # For basis='Z' the logical operator is the bottom row of data qubits
    # (smallest y), so its mask must occupy only the first grid row; for
    # basis='X' it is the left column, so the first grid column.
    circuit_z = make_circuit(circuit_name, 5, "Z")
    rows, _ = np.nonzero(circuit_z.observable_grid_masks[0])
    assert np.all(rows == 0)

    circuit_x = make_circuit(circuit_name, 5, "X")
    _, cols = np.nonzero(circuit_x.observable_grid_masks[0])
    assert np.all(cols == 0)
