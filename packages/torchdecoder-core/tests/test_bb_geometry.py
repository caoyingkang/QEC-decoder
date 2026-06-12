from functools import cache

import numpy as np
import pytest
import torch

from qecdec.circuits import BBCode_Circuit, CIRCUITS_REGISTRY
from torchdecoder_core.models import BBCodeGeometry


BATCH_SIZE = 7
SEED = 42

# The [[72,12,6]] code: A = x^3 + y + y^2, B = y^3 + x + x^2 on a 6x6 torus.
SMALL = dict(
    ell=6, m=6, a_poly=((3, 0), (0, 1), (0, 2)), b_poly=((0, 3), (1, 0), (2, 0))
)
BASES = ["Z", "X"]


@cache
def make_geometry(basis: str):
    circuit = BBCode_Circuit(**SMALL, basis=basis, rounds=3, error_rate=0.003)
    return circuit, BBCodeGeometry(circuit)


def test_rejects_non_bb_circuit() -> None:
    circuit = CIRCUITS_REGISTRY["RepetitionCode_Circuit"].with_uniform_error_rate(
        0.01, d=3, rounds=3
    )
    with pytest.raises(TypeError):
        BBCodeGeometry(circuit)


def test_rejects_unfiltered_circuit() -> None:
    circuit = BBCode_Circuit(
        **SMALL, basis="Z", rounds=3, error_rate=0.003, filter_detectors=False
    )
    with pytest.raises(ValueError):
        BBCodeGeometry(circuit)


@pytest.mark.parametrize("basis", BASES)
def test_adapter_matches_circuit_geometry(basis: str) -> None:
    circuit, geom = make_geometry(basis)
    assert geom.num_detectors == circuit.num_detectors
    assert geom.num_observables == circuit.num_observables
    assert (geom.num_layers, geom.ell, geom.m) == circuit.grid_shape

    assert geom.detector_sites.dtype == torch.int64
    assert np.array_equal(geom.detector_sites.numpy(), circuit.detector_grid_sites)
    assert geom.observable_masks.dtype == torch.bool
    assert np.array_equal(geom.observable_masks.numpy(), circuit.observable_grid_masks)
    assert geom.check_dataL_offsets.dtype == torch.int64
    assert np.array_equal(geom.check_dataL_offsets.numpy(), circuit.check_dataL_offsets)
    assert np.array_equal(geom.check_dataR_offsets.numpy(), circuit.check_dataR_offsets)


@pytest.mark.parametrize("basis", BASES)
def test_syndrome_grid_round_trip(basis: str) -> None:
    _, geom = make_geometry(basis)
    torch.manual_seed(SEED)
    syndromes = torch.randint(
        0, 2, (BATCH_SIZE, geom.num_detectors), dtype=torch.int32
    )

    grid = geom.syndrome_to_grid(syndromes)
    assert grid.shape == (BATCH_SIZE, geom.num_layers, geom.ell, geom.m)
    assert grid.dtype == syndromes.dtype
    assert torch.equal(geom.grid_to_syndrome(grid), syndromes)

    # Detectors fill the torus densely: every grid cell hosts one detector.
    assert geom.num_layers * geom.ell * geom.m == geom.num_detectors
    all_ones = torch.ones((1, geom.num_detectors), dtype=torch.int32)
    assert geom.syndrome_to_grid(all_ones).sum() == geom.num_detectors


@pytest.mark.parametrize("basis", BASES)
def test_grid_placement_matches_detector_sites(basis: str) -> None:
    """Detector i must land at its advertised (layer, row, col) grid site."""
    _, geom = make_geometry(basis)
    syndromes = torch.arange(geom.num_detectors, dtype=torch.int64).unsqueeze(0)
    grid = geom.syndrome_to_grid(syndromes)
    layers, rows, cols = geom.detector_sites.unbind(dim=1)
    assert torch.equal(
        grid[0, layers, rows, cols], torch.arange(geom.num_detectors)
    )


@pytest.mark.parametrize("basis", BASES)
def test_grid_round_trip_preserves_float_dtype(basis: str) -> None:
    _, geom = make_geometry(basis)
    torch.manual_seed(SEED)
    syndromes = torch.rand(BATCH_SIZE, geom.num_detectors)
    grid = geom.syndrome_to_grid(syndromes)
    assert grid.dtype == torch.float32
    torch.testing.assert_close(geom.grid_to_syndrome(grid), syndromes)


@pytest.mark.parametrize(
    "bad_shape",
    [(BATCH_SIZE,), (BATCH_SIZE, 3), (BATCH_SIZE, 3, 3)],
)
def test_syndrome_to_grid_rejects_bad_shape(bad_shape: tuple[int, ...]) -> None:
    _, geom = make_geometry("Z")
    with pytest.raises(ValueError):
        geom.syndrome_to_grid(torch.zeros(bad_shape))


def test_grid_to_syndrome_rejects_bad_shape() -> None:
    _, geom = make_geometry("Z")
    with pytest.raises(ValueError):
        geom.grid_to_syndrome(
            torch.zeros(BATCH_SIZE, geom.num_layers + 1, geom.ell, geom.m)
        )
