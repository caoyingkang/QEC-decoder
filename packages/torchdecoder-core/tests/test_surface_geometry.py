from functools import cache

import numpy as np
import pytest
import torch

from qecdec.circuits import CIRCUITS_REGISTRY
from torchdecoder_core.models import SurfaceCodeGeometry


BATCH_SIZE = 7
SEED = 42

CIRCUIT_CASES = [
    ("RotatedSurfaceCode_Phenom", 3, "Z"),
    ("RotatedSurfaceCode_Phenom", 5, "Z"),
    ("RotatedSurfaceCode_Phenom", 3, "X"),
    ("RotatedSurfaceCode_Circuit", 3, "Z"),
    ("RotatedSurfaceCode_Circuit", 5, "Z"),
]


@cache
def make_geometry(circuit_name: str, d: int, basis: str):
    circuit = CIRCUITS_REGISTRY[circuit_name].with_uniform_error_rate(
        0.01, d=d, rounds=d, basis=basis
    )
    return circuit, SurfaceCodeGeometry(circuit)


def test_rejects_non_surface_circuit() -> None:
    circuit = CIRCUITS_REGISTRY["RepetitionCode_Circuit"].with_uniform_error_rate(
        0.01, d=3, rounds=3
    )
    with pytest.raises(TypeError):
        SurfaceCodeGeometry(circuit)


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_adapter_matches_circuit_geometry(circuit_name: str, d: int, basis: str) -> None:
    circuit, geom = make_geometry(circuit_name, d, basis)
    assert geom.num_detectors == circuit.num_detectors
    assert geom.num_observables == circuit.num_observables
    assert (geom.num_layers, geom.grid_height, geom.grid_width) == circuit.grid_shape

    assert geom.detector_sites.dtype == torch.int64
    assert np.array_equal(geom.detector_sites.numpy(), circuit.detector_grid_sites)
    assert geom.detector_site_mask.dtype == torch.bool
    assert np.array_equal(geom.detector_site_mask.numpy(), circuit.detector_site_mask)
    assert geom.observable_masks.dtype == torch.bool
    assert np.array_equal(geom.observable_masks.numpy(), circuit.observable_grid_masks)


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_syndrome_grid_round_trip(circuit_name: str, d: int, basis: str) -> None:
    _, geom = make_geometry(circuit_name, d, basis)
    torch.manual_seed(SEED)
    syndromes = torch.randint(
        0, 2, (BATCH_SIZE, geom.num_detectors), dtype=torch.int32
    )

    grid = geom.syndrome_to_grid(syndromes)
    assert grid.shape == (
        BATCH_SIZE,
        geom.num_layers,
        geom.grid_height,
        geom.grid_width,
    )
    assert grid.dtype == syndromes.dtype
    assert torch.equal(geom.grid_to_syndrome(grid), syndromes)

    # Exactly the detector sites are populated; all other cells stay zero.
    all_ones = torch.ones((1, geom.num_detectors), dtype=torch.int32)
    assert geom.syndrome_to_grid(all_ones).sum() == geom.num_detectors


@pytest.mark.parametrize("circuit_name, d, basis", CIRCUIT_CASES)
def test_grid_round_trip_preserves_float_dtype(
    circuit_name: str, d: int, basis: str
) -> None:
    _, geom = make_geometry(circuit_name, d, basis)
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
    _, geom = make_geometry("RotatedSurfaceCode_Phenom", 3, "Z")
    with pytest.raises(ValueError):
        geom.syndrome_to_grid(torch.zeros(bad_shape))


def test_grid_to_syndrome_rejects_bad_shape() -> None:
    _, geom = make_geometry("RotatedSurfaceCode_Phenom", 3, "Z")
    with pytest.raises(ValueError):
        geom.grid_to_syndrome(
            torch.zeros(BATCH_SIZE, geom.num_layers + 1, geom.grid_height, geom.grid_width)
        )
